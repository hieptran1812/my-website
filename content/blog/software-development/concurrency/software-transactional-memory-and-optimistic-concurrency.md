---
title: "Software Transactional Memory and Optimistic Concurrency"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Treat memory like a database: wrap a block in atomically, read a snapshot, validate at commit, and get composable concurrency with no lock ordering and no deadlock."
tags:
  [
    "concurrency",
    "parallelism",
    "stm",
    "transactional-memory",
    "optimistic-concurrency",
    "composability",
    "clojure",
    "haskell",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-1.png"
---

Here is a bug that has cost real money in real banks. You have two correct functions. `withdraw(account, amount)` is correct: it locks the account, checks the balance, debits it, unlocks. `deposit(account, amount)` is correct: it locks the account, credits it, unlocks. Both pass their tests. Both have shipped. Now your product manager asks for a `transfer`, which any sane engineer writes as `withdraw(from, amount); deposit(to, amount)`. You ship it. Three weeks later the service freezes solid during a flash sale. The thread dump shows thread A holding the lock on account 17 and waiting for account 42, while thread B holds 42 and waits for 17. A `transfer(17, 42)` and a `transfer(42, 17)` ran at the same time and reached around each other into a circular wait — the textbook [deadlock](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them). Two correct operations, composed in the obvious way, produced an incorrect program.

That is the central failure of lock-based concurrency, and it has a name: **locks do not compose.** Correctness of `withdraw` plus correctness of `deposit` does not imply correctness of `withdraw` then `deposit`. To make the composition correct you have to reach inside both functions, learn which locks they take, and impose a global lock-ordering discipline across your whole codebase. That discipline is invisible in the type signature, unchecked by the compiler, and broken by the next person who adds a fourth account to the transfer. The abstraction leaks the moment you try to build on it.

![Two locked operations composed risk a deadlock while two STM transactions composed stay atomic with no ordering rules](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-1.png)

Software Transactional Memory — STM — fixes this by stealing the best idea in databases and pointing it at memory. You wrap a block of code in `atomically { ... }`. Inside, you read and write shared variables as if you were the only thread alive. The runtime runs the block **optimistically**: it reads a consistent snapshot, buffers every write in a private log, and at the end **validates** — did any value I read change while I was computing? If nothing changed, it **commits** the buffered writes in one atomic step. If something did change, it throws the buffer away and **retries** from a fresh snapshot. No locks are held across the block, so there is nothing to deadlock on. And the killer feature falls out for free: `transfer = atomically { withdraw(from); deposit(to) }` is automatically atomic, because two atomic blocks nested inside one atomic block are just one bigger atomic block. STM operations compose. By the end of this post you will be able to write a composable transfer in Clojure and Haskell, use `retry` and `orElse` to block correctly, recognize when a side effect inside a transaction is a time bomb, and decide — with numbers — when STM beats fine-grained locks and when it does not. This is the same shared-state-plus-scheduling hazard the [whole series](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) is about, tamed by a different mechanism: instead of establishing a happens-before order by holding a lock, STM establishes it by snapshot-and-validate.

## The one idea: optimistic read, validate, commit

Every concurrency-control scheme has to answer one question: when two threads touch the same data, how do we keep one from corrupting the other's view? There are exactly two strategies, and the entire field is a long argument about which to use where.

The **pessimistic** strategy assumes conflict is likely and prevents it up front. Before you touch the data, you acquire exclusive access — a lock — so nobody else can interfere. You pay the cost of coordination on every access, conflict or not. Mutexes, the `synchronized` keyword, database row locks: all pessimistic. "Lock first, then act."

The **optimistic** strategy assumes conflict is rare and detects it after the fact. You read the data without locking, do your work on a private copy, and at the very end you check whether anyone changed the data underneath you. If not, you publish your work. If so, you discard it and try again. You pay nothing on the common no-conflict path and pay a re-run only when a real conflict actually happened. "Act first, then validate." STM is the optimistic strategy applied to memory.

![Pessimistic locking acquires exclusive access before acting while optimistic STM reads a snapshot first and validates only at commit](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-5.png)

The mechanism inside an `atomically` block has five phases. **Begin**: the transaction starts and the runtime notes the current global state (in practice a global version clock, or per-variable version numbers). **Read**: every time your code reads a shared transactional variable, the runtime returns its value and records *which variable and which version* you saw — this growing record is the **read set**. **Compute**: when your code writes a shared variable, the runtime does *not* touch the real variable; it puts the new value in a private **write set** (the write log), and any subsequent read of that variable inside the same transaction returns the buffered value, so the transaction sees its own writes but no one else does. **Validate**: when the block ends, the runtime checks the read set — for every variable you read, is it still at the version you saw? **Commit or retry**: if the read set is still valid, the runtime atomically applies the write set to real memory and bumps versions; if any read is stale, it discards the write set and re-executes the block from a fresh snapshot.

![An STM transaction reads a snapshot, buffers writes, validates the read set at commit, then commits clean or retries on conflict](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-2.png)

The crucial property this buys you is **isolation**: within a single transaction, the world is frozen. You read a consistent snapshot — every value you see comes from the same logical instant — and nobody can observe your half-finished writes, because your writes live in a private buffer until the atomic commit. A transaction that reads ten variables sees ten values that were all simultaneously true at some point, never a torn mixture of old and new. That is exactly the guarantee you wanted from a lock, but you got it without ever holding a lock, and therefore without the deadlock, the priority inversion, and the lock-ordering bureaucracy that locks drag along.

Contrast that with the pessimistic side, where the cost is paid up front and unconditionally. A mutex makes every access — even one that never races anyone — go through an atomic acquire and release, a happens-before edge stamped on the memory even when there was no actual contention. Worse, the moment you hold *two* locks at once you have created a global ordering constraint that some other code path can violate, and that is the [deadlock](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them) we opened with. The optimistic side inverts the trade: nothing is paid on the common path where no conflict happens, and the entire cost lands only when a real conflict is detected — at which point you re-run. You are betting that conflicts are rare. When the bet is right (most concurrent code touching mostly-disjoint data), STM is both faster to reason about and competitive to run; when the bet is wrong (a stampede on one hot variable), the re-runs pile up and you pay more than a lock would have cost. The rest of this post is, in one form or another, an exploration of when that bet pays.

To make "validate the read set" concrete, here is the version-clock protocol (the TL2 scheme) stepped through as the runtime actually does it. At **begin**, the transaction reads the shared global version clock once and remembers it as `rv` (read version) — this is the timestamp of its snapshot. On every transactional **read** of variable `v`, the runtime first checks `v`'s version stamp is `≤ rv` (so you are reading a value that existed at your snapshot instant, not one written after you started); if it is newer, you have already lost and abort immediately. At **commit**, the runtime briefly acquires a lock on each variable in the write set (in a fixed address order, so this internal locking cannot deadlock), then atomically increments the global clock to get `wv` (write version), then **re-validates** every read-set variable's stamp is still `≤ rv`; if all pass, it writes the new values, stamps each with `wv`, releases the write locks, and the commit is done. If any read-set stamp is now `> rv`, some other transaction committed a write to something you read, so you release the write locks, discard everything, and retry with a fresh `rv`. The whole commit-time critical section is nanoseconds long and scoped to only the variables you are writing — which is why STM can hold "no locks across your code" and still commit atomically: the only locking is this brief, internal, ordered, deadlock-free gate.

#### Worked example: the lost update STM catches

Here is the canonical race the whole [series](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition) returns to — incrementing a shared counter — walked through STM's eyes. The counter is a transactional variable starting at `100`. Two transactions both run `atomically { c := c + 1 }`. The non-atomic hazard is the load-modify-store: both read `100`, both compute `101`, both store `101`, and one increment is lost.

Watch how STM forbids it. Transaction T1 begins, reads `c` and records "I saw `c` at version 7, value `100`" in its read set, computes `101`, buffers it. Transaction T2, interleaved, begins, reads `c` at version 7 value `100`, computes `101`, buffers it. Now T1 reaches commit: it validates — is `c` still at version 7? Yes. It commits, writes `101`, and bumps `c` to version 8. Now T2 reaches commit: it validates — is `c` still at version 7? **No, it is version 8 now.** Conflict detected. T2 discards its buffered `101`, re-runs the whole block from scratch: reads `c` at version 8, value `101`, computes `102`, validates (still version 8), commits `102`, bumps to version 9. Final value: `102`. No update lost. The optimistic re-run turned a silent corruption into a transparent retry. You wrote the obvious sequential-looking code and the runtime made it correct.

## The read set and the write set: how conflict is detected

The two logs are the whole machine, so let us make them concrete. A transaction is, at runtime, a pair of growing data structures plus the code that fills them.

The **read set** is the set of `(variable, version-you-observed)` pairs. Every transactional read appends to it. It is the transaction's claim about the state of the world: "my computation is only correct if all of these are still true at commit." The **write set** (or write buffer / redo log) is the set of `(variable, new-value)` pairs you intend to publish. Every transactional write appends to it instead of mutating live memory, and — this is the subtle part — every transactional read first checks the write set, so a write followed by a read of the same variable returns what you just wrote, keeping the transaction's view internally consistent.

Validation is a loop over the read set. In a **version-clock** scheme — used by Haskell's GHC STM and by the classic TL2 algorithm (Transactional Locking II, Dice, Shalev, and Shavit, 2006) — each transactional variable carries a version stamp, and there is a single shared global version counter. At begin, the transaction snapshots the global counter as its "read version." At commit it (briefly) locks the write-set variables, re-reads the global counter to get a "write version," and checks that every read-set variable's stamp is no newer than its read version. If all pass, it writes the new values, stamps them with the write version, and unlocks. The commit-time locks are held for nanoseconds, only over the variables being written, and are taken in a fixed address order so the commit itself cannot deadlock. This is why people sometimes say STM "uses locks internally" — it does, but only as a brief, internal, deadlock-free commit gate, never exposed to your code and never held while your block runs.

What counts as a conflict is precise: **a transaction conflicts only if a variable in its read set or write set was committed-over by another transaction during its lifetime.** Two transactions that touch entirely disjoint variables never conflict and both commit — STM gives you fine-grained, per-variable concurrency for free, without you partitioning anything into lock stripes by hand the way [readers-writer locks and lock striping](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity) force you to. Two transactions that both *read* the same variable but neither *writes* it also never conflict — read-read is not a conflict, exactly as in a database under snapshot isolation. Only a read-write or write-write overlap on the same variable triggers a retry.

It is worth laying out the four overlap cases explicitly, because the asymmetry between reads and writes is exactly what makes STM scale on read-heavy workloads and choke on hot writes. Two transactions whose variable sets overlap can overlap in one of four ways, and only two of them are conflicts:

| Overlap on a shared variable | Conflict? | Why |
| --- | --- | --- |
| both only read it | no | read-read; neither changes the version, both validate |
| one reads, one writes | yes | the writer bumps the version; the reader's read set goes stale |
| both write it | yes | one commits first and bumps the version; the second fails validation |
| neither touches it | no | disjoint; the variable is in no read set or write set |

This is precisely the conflict table of database snapshot isolation, which is the first hint at the deep equivalence we will make explicit later: STM detects conflicts the way an MVCC database does, by version, not by lock.

### The commit decision: validate, then branch

The whole optimistic gamble comes down to a single branch at the end of the block. When your code reaches the end of `atomically`, the runtime validates the read set, and the result is binary: either every variable you read is still at the version you saw — the read set is **clean** — or at least one was committed-over — there was a **conflict**.

![The commit decision validates the read set then branches to a clean commit or a conflict path that discards buffered writes and retries](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-4.png)

On the clean branch, the runtime publishes the write set to live memory in one atomic step, bumps the versions of the variables it wrote, and the transaction is done — its effects become visible to every other thread simultaneously, never half-applied. On the conflict branch, the runtime throws the write set away entirely (nobody ever saw it, so there is nothing to undo in shared memory) and re-executes the block from a fresh snapshot, building a brand-new read set and write set against the now-current state. This is why the retry is not a back-edge in the control flow that loops you back to where you were — it is a *fresh start* with a new snapshot, which is what makes the re-run correct rather than a replay of stale reads. The block is, semantically, a pure function from a snapshot to a write set, and the runtime is free to call that function as many times as it takes to land a commit on a snapshot nobody disturbed. Keep that "the body may run many times" fact in mind — it is the single most important thing to remember about STM, and it is the root of the no-side-effects rule we get to later.

#### Worked example: disjoint transfers run fully parallel

Concretely: account balances `A=500`, `B=300`, `C=900`, `D=100`, all transactional variables. Thread 1 runs `transfer(A→B, 50)`; its read set is `{A, B}`, its write set is `{A, B}`. Thread 2, at the same time, runs `transfer(C→D, 70)`; its read set is `{C, D}`, write set `{C, D}`. The two read sets are disjoint and the two write sets are disjoint. Neither transaction's validation finds any of its variables touched by the other, so **both commit on the first try, in parallel, with zero contention and zero retries.** Final: `A=450, B=350, C=830, D=170`. Compare the coarse-lock version — one global lock around all transfers — which would serialize these two perfectly independent operations and waste half your cores. STM extracted the available parallelism automatically because conflict detection is per-variable, not per-data-structure.

That last point is the quiet superpower. With locks, the granularity of your concurrency is the granularity of your locks, and choosing it is an agonizing manual trade-off: one big lock is simple but serializes everything; a thousand small locks parallelize well but deadlock and are a nightmare to get right. STM gives you the parallelism of a thousand fine locks with the programming model of one big lock. You write code as if there were a single global lock around every `atomically` block, and the runtime quietly discovers, transaction by transaction, that most of them do not actually conflict and lets them run together.

## Composability: the feature locks cannot give

Now we can state precisely what the intro promised. The defining property of STM, the one that locks structurally cannot provide, is **composability**: if `op1` is a correct atomic operation and `op2` is a correct atomic operation, then `atomically { op1; op2 }` is a correct atomic operation — automatically, with no new reasoning, no lock ordering, no possibility of deadlock.

Why does this work for STM and fail for locks? Because an STM transaction's atomicity is a property of the *dynamic extent* of the `atomically` block, not of any static lock object. When you nest two `atomically` blocks, a well-designed STM runtime **flattens** them: the inner blocks do not commit independently; they join the outer transaction, so all their reads land in one read set and all their writes in one write set, and there is a single validate-and-commit at the outermost boundary. The whole composed thing is one transaction. It is impossible for the composition to interleave with another thread between `op1` and `op2`, because there is no commit point between them.

Locks cannot do this because a lock's protection is scoped to the lock object, and two functions that each correctly take their own locks have no way to merge those scopes when you call them in sequence. Worse, the very thing that makes each one correct — acquiring a lock — is the thing that makes the composition deadlock, because now you are holding two locks at once and the acquisition order matters globally. The property you need (each function locks what it touches) directly causes the property you cannot tolerate (composing them creates lock-ordering constraints). There is no local fix. You would have to expose each function's lock set in its interface and have callers reason about global ordering — which is exactly the leaky, un-composable design we are trying to escape.

![A matrix comparing STM and locks on composability, deadlock risk, contention cost, and side effects](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-3.png)

Tim Harris, Simon Marlow, Simon Peyton Jones, and Maurice Herlihy made this the headline of their 2005 paper *Composable Memory Transactions*, and the example they used is the one worth internalizing. Take a bounded queue with `put` and `get`, each correctly synchronized. Suppose you now want "remove an item from queue 1 and insert it into queue 2, atomically, as a single indivisible action — and block until queue 1 has an item *and* queue 2 has space." With locks and condition variables this is genuinely hard: you cannot just call `get(q1)` then `put(q2)`, because between them another thread sees a state with the item nowhere; and the condition-variable waiting does not compose either, because each queue's [condition variable](/blog/software-development/concurrency/condition-variables-monitors-and-waiting-correctly) signals on its own monitor and you cannot wait on "q1 non-empty AND q2 non-full" without rewriting both queues' internals. With STM it is four lines and it just works. That is the whole pitch.

There is a deeper reason locks fail to compose that is worth stating plainly, because it shows the failure is structural and not a matter of being more careful. A lock-based component exposes a hidden part of its interface: *which locks it acquires, and in what order*. That information is not in the type signature, not in the docstring, and not enforced anywhere — but it is part of the contract, because to call two such components together safely you must reconcile their lock orders. So lock-based modularity is a lie: you cannot treat `withdraw` as a black box, because composing it with `deposit` requires you to open both boxes and learn their locking. The encapsulation does not survive composition, which is the working definition of a leaky boundary. STM repairs this because a transaction's atomicity is established by the runtime at the *outermost* `atomically` boundary, not by anything inside the functions — so the functions truly are black boxes, and you can compose them without knowing or caring what variables they touch. The information you needed to expose with locks (the lock set) is exactly the information STM lets you hide (because the runtime tracks the read/write sets for you). That is the modular-reasoning win, and it is why people who have lived through a lock-ordering refactor describe STM as a different category of tool, not a faster lock.

One more subtlety the composability story has to handle: what about *nested* transactions that conflict with each other or want to partially abort? The clean answer most production STMs use is **flattening** (also called closed nesting that subsumes into the parent): an inner `atomically` does not create an independent commit point; it merges its read and write sets into the enclosing transaction, so there is exactly one validate-and-commit at the top. A consequence is that an inner `retry` propagates out and blocks the *whole* composed transaction (which is what you want for `transferItem` — if q1 is empty, the whole move waits), and an inner abort rolls back the whole thing. Some STMs additionally support *nested* transactions with partial rollback for fine control, but the default and the one that gives you the simple composition story is flattening. The rule to keep is: `atomically` blocks nest like parentheses around one transaction, not like independent transactions in a sequence.

## retry and orElse: blocking transactions that compose too

So far STM handles *atomicity*. But real concurrent code also needs *blocking*: "wait until the queue is non-empty, then dequeue." A lock-based version uses a condition variable — wait on a monitor, get signaled. STM has something better and, crucially, composable: `retry`.

Inside a transaction you can call `retry`. Its meaning is: "the world is not in a state where I can make progress; abandon this transaction and re-run it — but do not busy-spin, **block me until something I read changes.**" This is the elegant part. The runtime already knows your read set. So when you `retry`, it parks the thread and watches exactly the variables in your read set; the instant any of them is committed-over by another transaction, it wakes you and re-runs the block. You never specify *what* to wait for — the read set *is* the wait condition, derived automatically. A `dequeue` that reads the queue, sees it empty, and calls `retry` will be woken precisely when some other transaction commits a write to that queue, and not a moment before, and never spuriously. You get condition-variable semantics with zero risk of a missed wakeup or a lost signal, because the wakeup set is computed from the data you actually touched.

```haskell
-- Haskell: a blocking dequeue using retry.
-- TVar is a transactional variable; the queue is two lists (front, back).
import Control.Concurrent.STM

dequeue :: TQueue a -> STM a
dequeue q = do
  contents <- readTQueue' q   -- reads the TVars; they go in the read set
  case contents of
    Just x  -> return x        -- got an item, commit normally
    Nothing -> retry           -- empty: block until a read-set TVar changes
```

The second combinator, `orElse`, is what makes blocking *composable*. `a orElse b` means: run transaction `a`; if `a` completes, you are done; **if `a` calls `retry`, do not block — instead discard `a`'s effects and run `b`; if `b` also retries, then block on the union of both read sets.** This is "try this, otherwise that" as a first-class operation on transactions. Now the impossible-with-locks example becomes trivial: to dequeue from `q1` *or else* from `q2`, whichever is ready first, you write `dequeue q1 orElse dequeue q2`. To wait until *either* of two conditions holds, you compose the two blocking transactions with `orElse`. There is no equivalent with condition variables — you cannot atomically `wait` on two different monitors' conditions and take whichever fires first without descending into a tangle of timeouts and re-checks.

```haskell
-- Atomically move an item from q1 to q2, blocking until q1 has an item.
-- Composes a blocking dequeue with an enqueue into ONE atomic action.
transferItem :: TQueue a -> TQueue a -> STM ()
transferItem q1 q2 = do
  x <- dequeue q1       -- blocks (retry) until q1 is non-empty
  enqueue q2 x          -- same transaction: the move is indivisible

-- Take from whichever queue is ready first; block only if BOTH are empty.
takeEither :: TQueue a -> TQueue a -> STM a
takeEither q1 q2 = dequeue q1 `orElse` dequeue q2
```

`retry` and `orElse` are the reason STM is not just "locks without deadlock" but a genuinely more expressive coordination model. Blocking and choice, the two things condition variables make painful and non-composable, become ordinary composable operations on transactions. This is the queue example from the *Composable Memory Transactions* paper made real, and it is the single most convincing demonstration that the composition story is not a slogan.

## The implementations: Clojure, Haskell, Scala

The idea is universal; the surface differs by language. Three production implementations are worth knowing, and they make different trade-offs about how strictly they wall off the dangerous parts.

![A matrix of STM implementations showing how each works and the niche it fits](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-6.png)

**Clojure** put STM at the center of its concurrency model. Clojure's whole philosophy is that values are immutable and *identity* is separate from *state*: a `ref` is a mutable reference cell whose succession of values you change only inside a transaction. You read a ref with `@` (deref) and you change it inside `dosync` using `alter` (apply a function: `(alter r f args)`), `ref-set` (set directly), or `commute` (a relaxed update for commutative operations that reduces retries). Because the values themselves are immutable, the snapshot a transaction reads can never be mutated out from under it — immutability and STM reinforce each other.

```clojure
;; Clojure: two accounts as refs; a composable, deadlock-free transfer.
(def account-a (ref 500))
(def account-b (ref 300))

(defn transfer [from to amount]
  (dosync                          ; begin an atomic transaction
    (alter from - amount)          ; buffered write to `from`
    (alter to   + amount)))        ; buffered write to `to`

;; transfer is itself atomic; calling it from a bigger dosync just flattens in.
(transfer account-a account-b 50)
;; @account-a => 450, @account-b => 350, never a torn intermediate state
```

Notice there is no lock, no ordering rule, and `transfer(a,b)` running concurrently with `transfer(b,a)` cannot deadlock — the worst that happens is one retries. Clojure's STM adds two refinements: `commute`, which lets commutative updates (like incrementing a counter or adding to a set) avoid conflicting with each other so they almost never retry, and `ensure`, which adds a variable to the read set to protect an invariant you read but do not write. Clojure also enforces the no-side-effects rule by convention and tooling rather than the type system — if you do I/O inside `dosync`, nothing stops you, and you will get burned (we will see exactly how below). Clojure's escape hatch for "do this side effect, but only once, after a successful commit" is `io!` blocks (which *throw* if executed inside a transaction, catching the mistake) and agents via `send`, whose actions are held and dispatched only on commit.

**Haskell** makes the guarantee airtight with the type system, and this is the most beautiful part of the whole story. A transactional variable is a `TVar a`. The only way to operate on `TVar`s is inside the `STM` monad, using `readTVar`, `writeTVar`, `newTVar`, plus `retry` and `orElse`. To actually run an `STM` action you wrap it in `atomically :: STM a -> IO a`, which turns a transaction into an `IO` action. Here is the magic: **`IO` actions cannot be run inside `STM`.** There is no function `IO a -> STM a`. So the type checker makes it *impossible to compile* a transaction that launches a missile, sends an email, or writes a file, because those are all `IO` and `IO` is not available in the `STM` monad. The single most dangerous mistake in STM — an irreversible side effect inside a block that may re-run — is a *compile error* in Haskell. That is the gold standard.

```haskell
-- Haskell: the same composable transfer, with TVars and atomically.
import Control.Concurrent.STM

transfer :: TVar Int -> TVar Int -> Int -> STM ()
transfer from to amount = do
  f <- readTVar from
  writeTVar from (f - amount)     -- buffered in the write set
  t <- readTVar to
  writeTVar to   (t + amount)

main :: IO ()
main = do
  a <- atomically (newTVar 500)
  b <- atomically (newTVar 300)
  atomically (transfer a b 50)    -- run the transaction; commit or retry
  -- transfer a b `andThen` transfer b c would compose into ONE transaction
```

If you tried to add `putStrLn "sent!"` inside `transfer`, GHC rejects it: `putStrLn` is `IO`, and `transfer`'s body is `STM`. You cannot even express the bug. The clean separation — pure functions, `STM` for coordinated mutable state, `IO` for effects — is enforced at the boundary, and STM sits exactly at the seam.

**Scala** offers STM as a library: ScalaSTM (`scala.concurrent.stm`), distilled from the earlier CCSTM and Akka's experimental STM. You wrap mutable state in a `Ref`, read/write inside `atomic { implicit txn => ... }`, and get `retry` and `orElse` too. Because Scala's type system does not have a Haskell-style effect monad walling off `IO`, ScalaSTM cannot *prevent* you from doing side effects in a block — but it gives you `Txn.afterCommit { ... }` to register an effect that runs exactly once, after a successful commit, which is the correct pattern and the one Haskell forces and Clojure's `io!`/agents encourage.

```scala
// Scala: ScalaSTM. Refs and an atomic block; afterCommit for safe effects.
import scala.concurrent.stm._

val accountA = Ref(500)
val accountB = Ref(300)

def transfer(from: Ref[Int], to: Ref[Int], amount: Int): Unit =
  atomic { implicit txn =>
    from() = from() - amount        // buffered write
    to()   = to()   + amount
    Txn.afterCommit { _ =>          // runs ONCE, only if the commit succeeds
      log.info(s"transferred $amount")
    }
  }
```

Three languages, one idea, a spectrum of enforcement: Haskell makes the safe path the only path that compiles; ScalaSTM gives you the safe tool and trusts you to use it; Clojure gives you the safe tools and a runtime tripwire (`io!`). The mechanism — optimistic read, validate, commit, retry — is identical in all three.

A word on the languages that *do not* ship STM, because the absence is informative. Go, Rust, Java, and C++ all chose locks, atomics, and (for Go) channels as their primary concurrency story, and none made STM a built-in. The reasons are consistent: STM's per-access bookkeeping is costly in a systems language that prizes predictable, allocation-free hot paths; STM's no-side-effects rule is unenforceable without an effect system, so in a language where any function can do I/O the safety guarantee evaporates; and the optimistic retry model interacts badly with the manual memory management and the "you pay for what you use" ethos of C++ and Rust. There are libraries — Rust has experimental STM crates, the JVM has Multiverse and ScalaSTM, GCC once shipped an experimental `__transaction_atomic` tied to TSX — but they remain niche. The honest summary is that STM thrives where the language already leans functional and immutable (Clojure, Haskell) so the no-side-effects rule and the snapshot model are natural, and stays marginal where mutation and effects are everywhere. Java's Project Loom is interesting here: it did *not* add STM, but its cheap virtual threads make the lock-and-block style cheaper, which is the pragmatic bet most mainstream runtimes made instead of optimism. Knowing which camp your language is in tells you whether STM is even on the table.

## Hardware transactional memory: Intel TSX and why it stalled

If STM in software is bookkeeping in a runtime, the obvious dream is to push that bookkeeping into silicon and make it nearly free. That dream is **Hardware Transactional Memory**, and Intel shipped a real version called **TSX** (Transactional Synchronization Extensions) starting with Haswell in 2013.

The mechanism is genuinely elegant. TSX co-opts the cache-coherence machinery you already have. You mark a region with `XBEGIN` / `XEND` (the RTM interface) or annotate a lock with `XACQUIRE` / `XRELEASE` (the HLE interface, designed to be a drop-in speedup for existing lock code). Inside the region, the CPU tracks every cache line you read and write as your transaction's read set and write set — at cache-line granularity, in L1. Your writes are held speculatively in the cache, not yet visible to other cores. The coherence protocol — [MESI](/blog/software-development/concurrency/cache-coherence-mesi-and-false-sharing), the same one that already broadcasts who owns each cache line — is repurposed as the conflict detector: if another core writes a line you read, or reads a line you wrote, the coherence traffic that announces it triggers an **abort**. On `XEND`, if no conflict occurred, the speculative writes commit atomically and become visible in one step. On abort, the CPU rolls back to the `XBEGIN` and jumps to a fallback path you provide. It is STM's exact algorithm, executed by the cache, at hardware speed, with no software read/write logs at all.

So why is HTM not how we all write concurrent code today? Three hard limits, and they are instructive because they are the same limits software STM has, just sharper.

**Capacity.** The read/write set lives in the cache, so a transaction that touches more lines than the cache (or the relevant cache way set) can track *must* abort — there is nowhere to record it. A loop over a large array, a transaction that calls into a lot of code, a region that spills L1: aborts, every time, with no conflict at all. HTM only works for *small* transactions, and "small" is a hardware constant you do not control.

**Unconditional aborts.** Many ordinary operations abort a TSX region no matter what: a syscall, a page fault, an interrupt, certain instructions (`CPUID`, `PAUSE` in some modes), context switches, and — critically — any I/O. So you can never make forward progress with HTM alone; you *must* always provide a non-transactional fallback (usually: take a real lock), which means HTM is best-effort speedup, not a programming model. The fallback path's lock is also what every transaction must read, so it serializes when contention is high.

**The bugs.** And then the part that actually killed it. TSX shipped with **errata**. In 2014 Intel disabled TSX in Haswell and early Broadwell parts via a microcode update because a bug could cause unpredictable behavior. Years later, side-channel research (TAA, "TSX Asynchronous Abort," CVE-2019-11135, a Meltdown-family transient-execution vulnerability) forced Intel to disable or recommend disabling TSX on a wide range of CPUs through microcode. By 2021 Intel had **deprecated and disabled TSX by default on most client and many server parts.** The hardware feature that was supposed to make transactions free spent most of its life either buggy, a security liability, or switched off. HTM is not dead in research — IBM POWER and z/Architecture have had robust HTM, and hybrid HTM-fast-path-with-STM-fallback designs exist — but on the commodity x86 most of us target, you cannot rely on it. The lesson: the read/write-set bookkeeping that STM does in software is doing real work, and you cannot just wish it onto the cache without inheriting capacity limits, abort storms, and a much bigger attack surface.

## STM is optimistic concurrency control — the database connection

If "read a snapshot, validate at commit, retry on conflict" sounds familiar, it should: it is precisely how a modern database runs transactions under **Multi-Version Concurrency Control** and snapshot isolation. STM is not *like* a database; STM *is* a database's concurrency-control algorithm, scoped to RAM and stripped of durability. Internalizing this mapping is the fastest way to understand STM, because everything you know about database isolation transfers directly.

![A matrix comparing STM in memory to database MVCC on unit, conflict detection, retry, and durability](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-7.png)

In Postgres or InnoDB under [MVCC](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb), a transaction reads from a consistent snapshot taken at its start — it sees a version of every row that was committed as of that instant, and concurrent writers create *new* versions rather than overwriting, so readers never block writers and writers never block readers. At commit, the database checks for conflicts; under serializable snapshot isolation it detects dangerous read-write dependency cycles and aborts one transaction, which the application is expected to retry. Map that onto STM term by term: the database's *row* is STM's *TVar*; the database's *snapshot at transaction start* is STM's *read-version clock*; the database's *MVCC version check at commit* is STM's *read-set validation*; the database's *serialization failure → retry* is STM's *conflict → re-run*. The algorithm is the same algorithm. STM took the textbook optimistic-concurrency-control protocol — the one Kung and Robinson formalized in 1981 — and applied it to in-memory variables instead of disk pages.

There is exactly one deep difference, and it is durability. A database transaction's commit is **durable**: it writes a [write-ahead log](/blog/software-development/database/write-ahead-log-how-databases-guarantee-durability) entry and `fsync`s so the change survives a crash. An STM transaction's commit is **volatile**: it just makes the new values visible in memory; if the process dies, everything is gone. This is why STM is fast (no disk, no `fsync`, commit is a few atomic memory operations) and why it is *not* a database (no recovery, no persistence). It is the "A," "C," and "I" of ACID — atomicity, consistency, isolation — without the "D." If you understand why your application has to wrap database writes in a retry loop to handle serialization failures, you already understand why STM transactions retry, because it is the identical mechanism. The same caveats apply, too: the [isolation level matters](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent), and a write-skew anomaly that snapshot isolation permits in a database has an exact analog in a naive STM transaction that reads two variables, checks an invariant across them, and writes only one.

The practical payoff of this equivalence is that two bodies of hard-won engineering wisdom merge into one. Everything the database community learned about optimistic concurrency over forty years — that read-mostly workloads love it, that hot-row write contention kills it, that you must make the application retry-safe and idempotent, that the *unit* you make atomic should be as small as correctness allows so the read set stays small and conflicts stay rare — transfers verbatim to STM. When you design an STM transaction, you are designing a database transaction that happens to live in RAM: keep it short, keep its read set small, do not do I/O inside it, make the surrounding code able to run it again, and expect to handle the rare-but-real retry. An engineer fluent in database transaction design is already fluent in STM design and just has to drop the durability concern. That is the most useful sentence in this whole post: STM is optimistic concurrency control for memory, so reason about it exactly as you reason about a database transaction.

#### Worked example: an invariant STM enforces that snapshot isolation would miss

Two accounts `A=100, B=100` with a business rule: the *combined* balance must stay non-negative, and either account may individually go negative as long as the sum is fine. Transaction T1 wants to withdraw `\$150` from A; transaction T2 wants to withdraw `\$150` from B. Each, in isolation, reads both balances (`A+B = 200 ≥ 150`, fine), and writes only its own account. Under plain snapshot isolation both see the old `200`, both think the rule holds, both commit: now `A=-50, B=-50`, combined `-100`, invariant violated. This is **write skew**. The STM fix is the same as the database fix: make the *read* of the variable you are checking but not writing part of the read set that gets validated. In Clojure you call `(ensure other-account)`; in Haskell you simply `readTVar` it (it is automatically in the read set) and the version check at commit catches the conflict; the second transaction sees the first's write to a variable it read, fails validation, and retries with the true new state. Same anomaly, same remedy, in memory and on disk. Understanding STM and understanding database isolation are the same skill.

## The costs and the no-side-effects rule

STM is not free, and an honest post says where it bleeds.

**Bookkeeping overhead.** Every transactional read and write goes through the runtime to be logged. A read is no longer a raw memory load; it is a load plus an append to the read set plus a write-set lookup (did I already write this?). A write is no longer a store; it is an append to the write set. A commit walks the entire read set to validate. This per-access tax — often a few times the cost of a raw access, plus allocation for the logs — is the price of optimism. For a transaction that touches three variables it is invisible; for one that scans a million-element structure it is brutal, and worse, a million-element read set is a million chances to conflict.

**Retry under contention.** Optimism is a bet that conflicts are rare. When that bet is wrong, you pay double: the wasted work of the aborted transaction *plus* the re-run, and possibly several re-runs. Under high write contention on a hot variable, transactions can **livelock** — each abort triggers a re-run that aborts again — and throughput collapses below what a simple lock would give, because a lock at least guarantees one winner makes progress while a naive STM can have everyone repeatedly abort. Good runtimes mitigate this with contention managers, backoff, and bounded-retry-then-fall-back-to-a-global-lock strategies, but the fundamental fact stands: **STM's performance is excellent under low contention and can be worse than locks under high contention.** This is the precise inverse of the pessimistic trade-off, and it is *the* number you must measure for your workload.

There is a subtler cost worth naming too: **inconsistent intermediate reads inside a doomed transaction.** Because a transaction runs optimistically against a snapshot that may already be stale, a long-running transaction can read values that, taken together, never formed a consistent state — for example reading variable X (still its old value) and Y (which another transaction has since changed but whose change your snapshot does not yet see). A *zombie* transaction like this is going to fail validation and retry anyway, so its final effect is harmless — but if its in-flight code does something irreversible *based on* those inconsistent reads (divides by a zero that the real consistent state never contained, follows a dangling pointer in a language without memory safety, or loops forever), you have a problem. Haskell sidesteps this because pure code on inconsistent values just produces a value that gets discarded; languages without memory safety need *opacity* guarantees (the runtime sandboxes a doomed transaction so it cannot observe an impossible state) to make zombies safe. It is one more reason STM is cleanest in a language that already forbids arbitrary effects.

**The no-side-effects rule.** This is the one that bites people, and it follows inevitably from "transactions may re-run." A transaction's body can execute *zero, one, or many times* before it commits — it is, semantically, a pure function from a snapshot to a write set, and the runtime may call it repeatedly. So any *irreversible side effect* inside the block — sending an email, charging a card, printing to a log, launching a missile, incrementing a Prometheus counter, writing a file — will happen *every time the block runs*, including on aborts that get rolled back. Memory writes are fine because the runtime buffers and can discard them. The outside world cannot be rolled back. An email, once sent, stays sent.

![A side effect inside a transaction sends three emails on retry while the same effect after commit sends one](/imgs/blogs/software-transactional-memory-and-optimistic-concurrency-8.png)

This is the BUG, in Clojure, that looks completely innocent:

```clojure
;; BUG: the email is a side effect INSIDE the transaction.
;; If this transaction retries (because account-a changed concurrently),
;; the email fires on EVERY attempt. Three retries => three emails.
(defn transfer-and-notify [from to amount email]
  (dosync
    (alter from - amount)
    (alter to   + amount)
    (send-email! email (str "Transferred " amount))))  ; runs on each retry!
```

Under contention this sends two, three, ten "your transfer is complete" emails for one transfer, and a confused customer files a ticket. The fix is to keep the transaction pure — only transactional memory operations inside — and perform the effect *exactly once, after a successful commit*, when you know the result is final:

```clojure
;; FIX: transaction is pure; the side effect runs once, after commit returns.
(defn transfer-and-notify [from to amount email]
  (dosync                              ; pure: only ref operations
    (alter from - amount)
    (alter to   + amount))
  ;; control reaches here only after the transaction COMMITTED (not on retries)
  (send-email! email (str "Transferred " amount)))  ; runs exactly once
```

In ScalaSTM you would write the effect inside `Txn.afterCommit { ... }`, which the runtime fires once on a successful commit. And in Haskell, as we saw, you *cannot write the bug at all* — `send-email!` is `IO`, the transaction body is `STM`, and the program will not compile until you move the effect outside `atomically`. The type system encodes the rule "no irreversible effects inside a possibly-re-run block" as a static guarantee. That is the single strongest argument for Haskell's STM: the most dangerous STM mistake is unrepresentable.

## A composable transfer, end to end

Let us pull the whole thesis into one worked example: a transfer that is composable, atomic, and free of lock ordering — the exact program that deadlocks under locks.

#### Worked example: transfer(a, b) and transfer(b, a) run concurrently and never deadlock

With locks, the deadlock scenario is: thread 1 runs `transfer(a, b)` and grabs lock-a, while thread 2 runs `transfer(b, a)` and grabs lock-b; now thread 1 waits for lock-b and thread 2 waits for lock-a — a circular wait, frozen forever. The standard lock fix is to *always acquire accounts in a fixed global order* (say, by account id), so both threads grab the lower id first. It works, but it is a non-local, error-prone discipline: every function that touches two accounts must know and obey it, and adding a third account to a transfer multiplies the care required.

With STM there is nothing to order. Here is the Haskell version, fully composable:

```haskell
import Control.Concurrent.STM
import Control.Concurrent (forkIO)
import Control.Monad (replicateM_)

-- A single, reusable, composable atomic operation.
transfer :: TVar Int -> TVar Int -> Int -> STM ()
transfer from to amount = do
  bal <- readTVar from
  if bal < amount
    then retry                    -- block until `from` has enough (composable!)
    else do
      writeTVar from (bal - amount)
      modifyTVar' to (+ amount)

main :: IO ()
main = do
  a <- atomically (newTVar 1000)
  b <- atomically (newTVar 1000)
  -- Two threads transfer in OPPOSITE directions, concurrently, repeatedly.
  _ <- forkIO $ replicateM_ 100000 (atomically (transfer a b 1))
  _ <- forkIO $ replicateM_ 100000 (atomically (transfer b a 1))
  -- No lock order. No deadlock possible. Conservation always holds.
  -- Final a + b == 2000, every run.
```

`transfer a b` and `transfer b a` running flat out against each other can never deadlock, because no lock is held across the block; the worst case is that one transaction's commit invalidates the other's read set and it retries. And note the `retry` in there: this transfer *blocks* until the source has enough funds, and that blocking *composes* — you can wrap two transfers in one `atomically` and the whole thing waits until both are satisfiable, something condition variables cannot give you without rewriting both accounts' internals.

Now the composition that is the entire point. A "two-way swap" — move `x` from a to b *and* `y` from b to a, atomically, all-or-nothing:

```haskell
-- Compose TWO transfers into ONE atomic transaction. No new locking logic.
-- Either both moves happen or neither does; no thread sees a half-done swap.
swap :: TVar Int -> TVar Int -> Int -> Int -> STM ()
swap a b x y = do
  transfer a b x     -- a -> b
  transfer b a y     -- b -> a, same transaction, one read set, one commit

-- atomically (swap a b 50 30)  -- indivisible; composes correct ops trivially
```

This is `transfer` reused, unchanged, composed into a larger atomic operation with zero new synchronization reasoning. The two correct operations compose into one correct operation — the promise from the first paragraph, delivered. Try writing `swap` with locks and you are back to global ordering across four lock acquisitions and a fresh chance to deadlock. In Clojure the same composition is `(dosync (transfer a b 50) (transfer b a 30))`, and in ScalaSTM `atomic { implicit t => transfer(a,b,50); transfer(b,a,30) }`. Same idea, same freedom, three languages.

## Measured: STM versus fine-grained locks under contention

Now the honest part. STM's marketing is composability; its reality is a performance curve that depends entirely on contention, and you must know the shape or you will deploy it in exactly the case where it loses. Here is the behavior to expect, drawn from the published STM literature (the Haskell STM benchmarks, the TL2 paper, and the broad body of STM-versus-locks studies) and stated as order-of-magnitude trends, not as numbers from a specific machine — measure your own.

The independent variable is **contention**: roughly, the probability that two concurrently-running transactions touch an overlapping variable. The metrics are **throughput** (committed transactions per second) and **retry rate** (aborts per commit). The qualitative result is consistent across studies:

| Contention level | STM behavior | Fine-grained locks | Winner |
| --- | --- | --- | --- |
| Low (disjoint data) | near-zero retries; scales with cores | needs many locks; lock overhead per op | STM (simpler, scales) |
| Moderate | some retries; throughput still climbs | some lock contention; risk of deadlock bugs | roughly even; STM far simpler to get right |
| High (one hot variable) | high retry rate; throughput can collapse | serializes on the hot lock but each op progresses | locks (STM wastes work re-running) |
| Read-mostly | reads never conflict; excellent | reader-writer lock helps but writers still block | STM (read-read is free) |

The single most important point: **STM's abort rate grows with contention, and aborted work is pure waste.** At low contention STM gives you the scalability of fine-grained locking with the programming simplicity of one global lock — the best of both, and the case it was built for. At high contention on a single hot variable, every transaction touching it keeps getting its read set invalidated, the retry rate climbs toward "most attempts abort," and throughput can fall *below* a single coarse mutex, because the mutex at least lets one winner finish while STM has everyone repeatedly redo and discard work.

#### Worked example: a counter under 8 threads, two designs

Picture eight threads, each doing nothing but `increment(shared_counter)` ten million times — maximal contention on one variable. The fine-grained-lock design is a mutex around the counter: every increment takes the lock, increments, releases; the threads serialize on that one lock, so you get roughly single-threaded throughput plus lock overhead, but you *do* get steady forward progress — eighty million increments happen at the rate one core can take-lock-increment-release, with no wasted work. The naive STM design wraps each increment in `atomically`: every transaction reads the counter (version N), computes N+1, and at commit most of them discover the counter is now version N+5 because seven other threads committed in the meantime — so they abort and retry, and the abort rate soars; you may see two, three, or more aborts per successful commit, meaning the machine does several times the work to land eighty million increments, and aggregate throughput drops *below* the simple mutex. The honest engineering response is not "STM is bad" but "this is STM's worst case — one hot variable, write-heavy — and you should either not use STM here, or use a contention-aware primitive (Clojure's `commute` for a commutative increment, which is designed precisely so concurrent increments do not conflict and almost never retry, recovering the parallelism)."

How to **measure this honestly** for your own workload: warm up the JVM/runtime (STM's first runs pay JIT and allocation costs); run many iterations and report a distribution, not one number, because retry counts are nondeterministic and scheduler-dependent; sweep the thread count from 1 to past your core count and *find where the curve bends down*, because that knee is the answer; report the retry rate alongside throughput (a high-throughput STM run with a 0.01 abort rate and a low-throughput one with a 5.0 abort rate are completely different situations and the throughput number alone hides it); and name your platform, because the cache, the core count, and the memory model all move the result. Never quote a single "STM is 2× faster" or "2× slower" number without the contention level attached — it is meaningless without it.

There is a simple analytic way to see why the abort rate climbs the way it does, and it is worth a sentence of math because it tells you the *shape* of the curve. Suppose `n` threads each run a short transaction that touches a hot variable, and a transaction's commit takes time roughly proportional to its read-set walk. The probability that a given transaction's window overlaps a conflicting commit grows with the number of *other* threads racing for the same variable — to first order, the conflict probability scales like the fraction of time the variable is "in flight" times `(n-1)`. So the expected number of attempts per successful commit grows roughly linearly with `n` once the variable is hot, which means the *useful* work per unit time (commits, not attempts) flattens and then declines as more threads pile on, while a lock holds throughput flat at one-winner-at-a-time. That is the precise reason a write-contended STM curve bends down where a lock's plateaus: aborted attempts are work the machine does and then discards, and their count grows with the crowd. The fix is never "tune the STM harder" — it is "reduce the contention," by splitting the hot variable, using a commutative update that does not conflict, or stepping outside STM for that one cell.

#### Worked example: choosing the design from the contention number

Suppose you measure your real workload and find that, across a representative trace, fewer than 1% of transaction pairs touch an overlapping variable, and most transactions read several variables and write one or two. That low overlap is STM's home turf: you will see a near-zero abort rate, the code stays a simple `atomically` block per operation, you pay no lock-ordering tax, and throughput scales with cores. Pick STM. Now suppose instead you measure that 40% of transactions hammer one shared sequence-number variable that every operation must bump. That is the opposite world: STM will abort constantly on that one variable, and the right move is to take the sequence number *out* of STM entirely — make it a single atomic fetch-add (one `cmpxchg`, no read set to invalidate) and leave only the genuinely-multi-variable coordination inside `atomically`. The decision is not ideological; it is read straight off the contention measurement. The whole skill of using STM well is measuring overlap and keeping the hot, write-contended cells out of the transactional path.

## Case studies / real-world

**Haskell STM in production.** The strongest production evidence is Haskell's, because STM is built into GHC's runtime and used in real systems. The *Composable Memory Transactions* paper (Harris, Marlow, Peyton Jones, Herlihy, PPoPP 2005) is the canonical reference, and GHC's STM has run in production web services and concurrent infrastructure for nearly two decades. The reported wins are exactly the ones this post argues: complex coordination logic (multi-resource blocking, "wait for A or B," atomic moves between structures) that would be a deadlock-prone tangle with locks becomes short, composable, and correct-by-construction with `atomically`, `retry`, and `orElse`. The type-enforced no-IO-in-STM rule is repeatedly cited as the reason teams trust it: the most dangerous mistake cannot be written. Simon Marlow's book *Parallel and Concurrent Programming in Haskell* documents real STM patterns (a concurrent network server, a bank, a windowing system) with measured behavior.

**Clojure's identity-and-state model.** Rich Hickey designed Clojure (2007–) around the explicit thesis that *state* is a succession of immutable *values* over time, and that coordinated change to shared state belongs in transactions, not locks. Clojure shipped `ref`/`dosync`/`alter` as a first-class language feature, alongside atoms (for uncoordinated single-variable change) and agents (for asynchronous change). The practical lesson the Clojure community internalized — and that you should take — is that STM is the right tool for the *minority* of state that needs *coordinated* multi-variable atomic updates, and that most concurrent state is better served by the simpler atoms; STM is a precision instrument, not a default. The `commute` primitive in Clojure is the standard production answer to the hot-counter contention problem above. The deeper architectural lesson is the separation Clojure makes structural: by forcing values to be immutable, the snapshot a transaction reads can never be mutated out from under it, so the STM only has to coordinate the *swapping of references*, not guard against in-place mutation of the data those references point to. That is why Clojure's STM is simpler than an STM bolted onto a mutate-in-place language: half the hazard (aliased mutable structure changing during a transaction) is designed away before the transaction even starts. The principle generalizes beyond Clojure — immutability and optimistic concurrency reinforce each other, which is why both functional STM implementations lean on it.

**Where STM did *not* win, and why that is honest.** It is worth being candid that STM did not become the dominant concurrency model the early-2000s enthusiasm predicted. Mainstream systems languages stuck with locks, atomics, and channels; hardware TM was retired on x86; and even in functional languages STM is reserved for the coordinated-state minority. The reason is not that the idea is wrong — it is that the idea has a narrow, real cost envelope (per-access overhead, write-contention aborts, the no-effects rule) and the industry mostly had problems that fit cheaper tools, or had effects everywhere so the safety guarantee did not hold. The correct takeaway from STM's trajectory is not "STM failed" but "STM is a specialist": when your problem is genuinely multi-variable atomic coordination that must compose, STM is unmatched and you should reach for it; when it is not, the simpler tool wins, and a good engineer can tell the difference by looking at the shape of the shared state and the contention number.

**The TSX errata and retirement.** Intel TSX is the cautionary tale, and it is well documented. Intel disabled TSX in Haswell and early Broadwell via a 2014 microcode update after discovering an erratum that could cause unpredictable behavior. The TAA side-channel vulnerability (TSX Asynchronous Abort, CVE-2019-11135, disclosed November 2019) — a transient-execution flaw in the Meltdown/MDS family — forced microcode mitigations that disabled TSX on many parts, and Intel subsequently deprecated and disabled TSX by default across most client and many server CPUs by 2021. The hardware that promised free transactions spent its life buggy, then a security liability, then switched off. The durable lesson for an engineer: the read/write-set bookkeeping software STM does is real, necessary work, and the cache cannot simply absorb it for free without inheriting capacity limits, abort storms, and a wider attack surface — which is why software STM, slower but reliable, is what actually shipped and stuck.

## When to reach for this (and when not to)

STM is a precision tool with a sharply-shaped trade-off. Be decisive about it.

**Reach for STM when you need composability of atomic operations across multiple variables.** This is its irreplaceable strength. If your problem is "atomically update several pieces of shared state, and build bigger atomic operations out of smaller ones, without a lock-ordering discipline," nothing else is as clean. The multi-account transfer, the atomic move between two queues, the "wait until A *and* B are ready" coordination — STM makes these trivial and lock-based code makes them deadlock-prone. If you find yourself drawing a lock-ordering diagram or have already been bitten by a lock-ordering deadlock, STM is the answer.

**Reach for STM when the workload is read-mostly or low-contention.** Read-read never conflicts, so a read-heavy structure under STM scales beautifully with near-zero retries, and you get fine-grained concurrency without hand-partitioning anything into lock stripes. This is STM's performance sweet spot.

**Do not use STM under high write contention on a hot variable.** This is its worst case: the retry rate climbs, work is wasted on aborts, and throughput can fall below a plain mutex. If your bottleneck is many threads hammering one counter or one cell, use a lock (one winner makes progress), a contention-aware primitive (Clojure `commute`, or an [atomic fetch-add](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) if it is a single variable), or a [lock-free](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads) structure — measure the contention first.

**Never put an irreversible side effect inside a transaction.** No I/O, no network calls, no emails, no logging-that-matters, no metrics increments, no anything-the-world-remembers — because the block may re-run any number of times. Keep the transaction pure (memory only) and do the effect once, after commit (`Txn.afterCommit`, an effect after `dosync` returns, or — in Haskell — the type checker forcing it out). If you cannot avoid the effect being coupled to the decision, STM is the wrong model; use a lock you hold across the effect, or restructure so the transaction only *decides* and a post-commit step *acts*.

**Do not reach for STM if a single atomic or a single lock already solves it.** STM's bookkeeping is overkill for one variable (use an atomic) or for a short critical section with no composition need (use a mutex). STM earns its overhead only when you need *coordinated, composable, multi-variable* atomicity. And remember that STM gives you A, C, and I but not D — if you need durability, you need an actual database, and the [transaction isolation](/blog/software-development/database/isolation-levels-and-the-anomalies-they-prevent) story there is the same algorithm with `fsync` added.

## Key takeaways

1. **Locks do not compose; STM does.** Two correct lock-based operations can deadlock when sequenced; two correct STM transactions nest into one correct transaction with no lock ordering and no deadlock. Composability is STM's defining feature.
2. **STM is optimistic concurrency control for memory.** Read a consistent snapshot, buffer writes in a private log, validate the read set at commit, commit if clean or retry if a read changed. No locks are held across your block.
3. **Conflict is per-variable, read-write or write-write.** Disjoint transactions run fully parallel; read-read never conflicts. You get fine-grained concurrency with a one-big-lock programming model.
4. **`retry` and `orElse` make blocking composable.** `retry` blocks you until a read-set variable changes (the read set *is* the wait condition); `orElse` is first-class "try this, else that" on transactions — both impossible to compose with condition variables.
5. **STM is a database without the D.** It gives atomicity, consistency, and isolation via the exact MVCC/snapshot-isolation algorithm, minus durability — which is why it is fast and why it is not a database. Write skew exists in both; the fix (validate the variable you read) is the same.
6. **Never do irreversible side effects inside a transaction.** The block may run many times before it commits, so any I/O fires on every retry. Keep transactions pure; perform effects once, after commit. Haskell's type system makes this a compile error.
7. **STM wins at low contention and read-mostly; it loses at high write contention.** Aborted work is pure waste, so a hot variable can drive throughput below a plain mutex. Measure contention and the retry rate, not just throughput.
8. **Hardware TM is not a reliable substitute.** Intel TSX had the right algorithm but capacity limits, unconditional aborts (any I/O, syscall, fault), errata, and the TAA vulnerability got it disabled across most x86 — software STM is what actually ships.

## Further reading

- **Tim Harris, Simon Marlow, Simon Peyton Jones, Maurice Herlihy — *Composable Memory Transactions* (PPoPP 2005).** The foundational STM paper; `retry`/`orElse` and the composability argument come from here. Read this first.
- **Simon Marlow — *Parallel and Concurrent Programming in Haskell* (O'Reilly).** The STM chapters give runnable `TVar`/`atomically`/`retry`/`orElse` code with measured behavior and real examples (bank, network server, windowing).
- **Maurice Herlihy & J. Eliot B. Moss — *Transactional Memory: Architectural Support for Lock-Free Data Structures* (ISCA 1993).** The paper that started hardware TM; the lineage behind Intel TSX.
- **Dave Dice, Ori Shalev, Nir Shavit — *Transactional Locking II* (DISC 2006).** The TL2 algorithm — the global-version-clock validation scheme this post describes in detail.
- **H. T. Kung & John T. Robinson — *On Optimistic Methods for Concurrency Control* (ACM TODS 1981).** The original optimistic-concurrency-control paper; STM is this algorithm applied to memory.
- **Rich Hickey — *Clojure's approach to Identity and State* (clojure.org).** The design rationale for `ref`/`dosync` and the separation of identity from value.
- **Intel — TSX documentation and the TAA / CVE-2019-11135 advisory.** Primary sources for the hardware TM capacity limits, abort conditions, and the security-driven retirement.
- **Within this series:** the [intro to why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), the [deadlock conditions](/blog/software-development/concurrency/deadlock-the-four-conditions-and-how-to-break-them) STM sidesteps, the [ABA and torn-read hazards](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads) optimism must respect, the database [MVCC deep dive](/blog/software-development/database/mvcc-deep-dive-postgres-vs-innodb) STM mirrors, and the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model) for choosing a model.
