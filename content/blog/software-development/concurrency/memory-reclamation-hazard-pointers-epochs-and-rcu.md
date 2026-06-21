---
title: "Memory Reclamation: Hazard Pointers, Epochs, and RCU"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The hard part of lock-free that nobody warns you about: in a structure with no locks, deciding when it is finally safe to free a node another thread might still be reading."
tags:
  [
    "concurrency",
    "parallelism",
    "lock-free",
    "hazard-pointers",
    "rcu",
    "epoch-reclamation",
    "memory-management",
    "use-after-free",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-1.png"
---

You build your first lock-free stack. The push is a thing of beauty: read the head, point your new node at it, `compare-and-swap` the head to your node, retry on failure. The pop looks symmetric: read the head, read `head->next`, `compare-and-swap` the head to `next`, return the old head. You run it on one thread and it is correct. You run it on eight threads and it is correct — for a while. Then, under load, at 3 AM, it segfaults. The core dump points at a line that reads `head->next`. The pointer was not null. It was a valid address an instant ago. By the time the load executed, that memory had been freed and handed to another allocation, and you read garbage — or you read into an unmapped page and the kernel killed you.

This is the bug that nobody warns you about when they teach `compare-and-swap`. The CAS loop is the famous part, the part in every tutorial. The unglamorous, genuinely hard part is the question that comes *after* you successfully pop a node: **when is it safe to call `free` on it?** In a structure protected by a lock, the answer is trivial — nobody can be inside the structure while you hold the lock, so once you have removed a node and dropped the lock, it is yours. In a lock-free structure there is no lock, which is the whole point, and so there is no moment when you can be sure no other thread is still holding a pointer into the node you just removed. Another thread read the head a nanosecond before you unlinked it. It is now sitting on a pointer to a node you are about to free, and it is about to dereference it.

![timeline showing thread one reading a node while thread two unlinks and frees it then thread one dereferences freed memory and crashes](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-1.png)

This problem is **safe memory reclamation**, and it is the reason lock-free programming is hard in languages without a garbage collector. If you write your stack in Java or Go or C#, the runtime tracks every live reference and will not free a node while any thread can still reach it; the problem evaporates because the GC is, in effect, a universal reclamation scheme running underneath you. In C, C++, or Rust without a GC, you own the memory, and you have to answer the "when is `free` safe" question yourself. The answer is a *reclamation scheme*: a protocol every thread participates in so that a removed node is freed only once it is provably unreachable. This post is about the three schemes that matter — **hazard pointers**, **epoch-based reclamation**, and **read-copy-update (RCU)** — what each one costs, when each one wins, and how to wire one into the lock-free stack and queue from the [building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures) post. By the end you will be able to take a lock-free pop that crashes and make it safe, and to choose the scheme that fits your access pattern instead of cargo-culting one.

If you have not yet internalized *why* concurrency forces these questions on you at all, the series [intro on why concurrency is hard](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) is the place that frames the whole thing: shared mutable state plus nondeterministic scheduling equals the hazard. Reclamation is that hazard at its most pitiless — the window between "I removed it" and "I freed it" is exactly a place where the scheduler can interleave another thread, and the consequence is not a wrong answer but a corrupted heap.

## The bug, walked instruction by instruction

Let us be precise about what goes wrong, because the precision is the whole point. Here is a lock-free Treiber stack pop in C, written naively, with no reclamation scheme at all:

```c
struct node { int value; struct node *next; };
_Atomic(struct node *) head;

int pop(int *out) {
    struct node *old_head, *new_head;
    do {
        old_head = atomic_load(&head);          // (1)
        if (old_head == NULL) return 0;         // empty
        new_head = old_head->next;              // (2)  <-- the danger
    } while (!atomic_compare_exchange_weak(&head, &old_head, new_head)); // (3)
    *out = old_head->value;
    free(old_head);                             // (4)  <-- the danger
    return 1;
}
```

Read line (2) carefully. Between line (1), where T1 loads `head` into `old_head`, and line (2), where T1 dereferences `old_head->next`, the OS scheduler is free to suspend T1 and run T2. Suppose T2 is also popping. T2 successfully pops the very same node T1 is looking at — T2's CAS at line (3) succeeds, T2 returns the node, and T2's line (4) calls `free(old_head)`. Now the scheduler resumes T1. T1 executes line (2): it dereferences `old_head`, a pointer to memory that has been freed and may already have been reallocated and overwritten by some unrelated `malloc` elsewhere in the program. T1 reads a `next` value that is no longer the stack's `next` — it is whatever bytes now live there. This is a **use-after-free**.

There are two distinct ways this kills you. The benign-looking one: T1 reads garbage into `new_head`, its CAS at (3) fails (because `head` has changed), and it loops. No crash — but if the freed memory happened to be reallocated such that its first eight bytes again equal the old `head` value, T1's CAS can *succeed* with a stale, wrong `new_head`, corrupting the stack silently. This is the [ABA problem](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads) wearing its other face: ABA is not just about a pointer value being reused, it is fundamentally enabled by the fact that the node was freed and reused. The malignant one: T1 dereferences a pointer into an unmapped page and the kernel delivers `SIGSEGV`. Either way the root cause is the same — **T1 held a pointer into a node that T2 freed.**

#### Worked example: the exact interleaving that frees out from under you

Take a stack holding two nodes, A on top of B (`head -> A -> B -> NULL`). Two threads pop concurrently. Trace the steps with explicit instruction ordering:

1. T1 executes (1): `old_head = A`.
2. T1 executes (2): reads `A->next`, gets `B`. So T1's `new_head = B`. T1 is now suspended.
3. T2 executes (1): `old_head = A`. T2 executes (2): `new_head = B`. T2 executes (3): CAS `head` from A to B — succeeds. `head -> B` now. T2 returns A and executes (4): `free(A)`.
4. T2 pops again: pops B, frees B. The stack is empty.
5. The allocator hands A's freed block to an unrelated `malloc` call elsewhere, which writes its own data into it. The first eight bytes of A's old storage are now, say, the bytes `0x4141...`.
6. T1 resumes at (3): CAS `head` from `old_head=A` to `new_head=B`. But `head` is now NULL (or `B`, depending on timing). The CAS *fails*, T1 loops back to (1), reloads `head=NULL`, returns empty. Lucky — no crash this time.

Now run the same trace but suspend T1 *between* (1) and (2) instead, so T1 reads `A->next` *after* A is freed at step 3. At that point `A->next` is `0x4141...` garbage. T1's `new_head` is garbage. If `head` still happens to equal A (because some later push reused A's address — classic ABA), T1's CAS succeeds and installs `0x4141...` as the head. The next pop dereferences `0x4141...` and the process dies. The window is a single instruction wide, and it opens on every pop. That is why "it works under light load and crashes under heavy load" — heavy load is just more chances to land in the window.

The fix is not to make the CAS smarter. The CAS is fine. The fix is to never `free` A while any thread might still dereference it. That is a reclamation scheme, and the rest of this post is the three good ways to build one.

## Why reclamation is the hard part of lock-free

It is worth dwelling on *why* this is genuinely the hard part, harder than the CAS loop itself, because the difficulty is structural and not incidental.

A lock gives you a **mutual-exclusion boundary**: a clean before-and-after where, while you hold the lock, the data structure is yours alone. Removal and reclamation can happen inside the same critical section. You unlink the node, you free it, you drop the lock — and because nobody else could enter while you held the lock, nobody else could have grabbed a pointer to the node you removed. The lock does double duty: it serializes the *structural* mutation and it serializes the *lifetime* decision. The [mutual exclusion post](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) shows this boundary in detail; here the point is just that with a lock, lifetime is free.

Lock-free programming deliberately removes that boundary. There is no critical section. Threads enter and traverse the structure with no coordination beyond atomic loads and stores. That buys you the lock-free progress guarantee — no thread can block the whole system by holding a lock and getting suspended — but it costs you the lifetime boundary. The instant you remove a node, you have to confront the fact that you have no idea who else is currently pointing at it. There is no list of "threads currently inside the structure." There is no lock whose release marks "everyone is out." The removal is a *local* atomic operation; the question "is anyone still reading this?" is a *global* one. Reclamation is precisely the work of reconstructing a global liveness answer without a global lock.

This is why the three schemes look different but are doing the same thing: they each reconstruct, by a different protocol, the fact a lock would have given you for free — **a moment at which no thread can possibly hold a pointer to the retired node.** Hazard pointers reconstruct it by having every thread *publish* the pointers it is using, so the reclaimer can check. Epoch-based reclamation reconstructs it by having threads *announce an epoch*, so the reclaimer can wait for everyone to advance. RCU reconstructs it by defining a *grace period* — an interval after which every reader that could have held the old pointer has provably finished. Three protocols, one job: find the moment that `free` becomes safe.

There is a second reason it is hard, beyond the conceptual one: the schemes interact with the *memory model*. Publishing a hazard pointer is a store; checking it is a load; for the protocol to be correct, those have to be ordered with the right fences, or the reclaimer can fail to observe a hazard that was set "before" the free in program order but became visible "after" it on the hardware. Get the [acquire-release ordering](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences) wrong and the scheme silently fails to protect, reintroducing exactly the use-after-free it exists to prevent. So reclamation is not just an allocator problem; it is a memory-model problem wearing an allocator's clothes.

## Why garbage-collected languages dodge it entirely

Before we build a scheme by hand, it is worth seeing clearly why some languages never make you think about this — because understanding *what the GC gives you* tells you exactly *what your hand-rolled scheme must reproduce.*

In a tracing-GC language, you never call `free`. You drop your reference to a popped node and move on. The collector periodically (or concurrently) traces from the roots — stacks, registers, globals — and reclaims anything unreachable. The crucial property: **the collector will not reclaim a node while any thread can still reach it through a live reference.** A thread that is mid-pop and holding a pointer to node A is, by definition, holding a live reference to A; the collector sees it (on the stack, in a register, wherever) and refuses to collect A. Your use-after-free is impossible because "use" implies "reachable" implies "not collected."

That is exactly the guarantee a manual scheme must reconstruct. The GC has an advantage you do not: it can stop the world (or use read/write barriers) to get a globally consistent view of every reference every thread holds. Your manual scheme cannot stop the world without reintroducing a global pause that defeats the purpose. So manual schemes are, in a sense, *specialized, cooperative, partial GCs* — they track only the pointers into one structure, only the ones threads explicitly publish, and they reclaim only retired nodes. They trade the GC's generality (it tracks every object) for far lower and more predictable cost (they track a handful of hazard slots or a single epoch counter).

This framing also explains a real engineering decision: **if you already have a good concurrent GC, you usually should not hand-roll reclamation.** The JVM's lock-free structures in `java.util.concurrent` — `ConcurrentLinkedQueue`, `ConcurrentSkipListMap` — simply drop references and let G1 or ZGC handle lifetime. They are correct and fast and contain zero hazard-pointer code. The manual schemes earn their complexity only in GC-free languages, or in the few GC'd contexts where you are managing off-heap or native memory the collector does not see. Keep that in your back pocket for the "when to reach for this" section; it is the single most common mistake — reaching for hazard pointers in a language that would have done it for you.

Python is an instructive middle case. CPython uses reference counting plus a cycle collector, and its lock-free story is dominated by the [GIL](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs), which serializes bytecode execution and makes most of this moot for pure-Python data structures. When you genuinely need lock-free reclamation in a Python context you are almost always in a C extension, back in C's world with C's rules. So we will not write Python here; the [python-performance series](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) owns that story, and the honest answer for pure Python is "the GIL and refcounting already did it." This post lives in the manual-memory languages where the problem is real.

## Hazard pointers: publish what you use, defer the free

The hazard-pointer scheme, introduced by Maged Michael in 2004, is the most conservative and the most predictable of the three. The idea is direct: **before a thread dereferences a shared pointer, it publishes that pointer in a per-thread, single-writer/multi-reader slot called a hazard pointer. A node is freed only when no hazard pointer references it.**

![before and after comparison showing no reclamation scheme causing a use-after-free versus a hazard pointer keeping a referenced node alive](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-3.png)

Each thread owns a small fixed array of hazard slots — typically one or two per structure (a stack pop needs one; a list traversal that holds a current and a next needs two). The slots are atomic pointers, written only by their owning thread but readable by all. When a thread wants to dereference `head`, it does a careful dance:

1. Read `head` into a local `p`.
2. Publish `p` into its hazard slot with a store, then a *fence* so the store is visible.
3. **Re-read `head`.** If it still equals `p`, then `p` was protected from the moment it was published, and it is now safe to dereference. If it changed, loop — some other thread may have removed and is retiring `p`, and your publication came too late, so you must restart.

That re-read is the whole correctness argument and the part people get wrong. Publishing the hazard pointer is not enough; you must verify that the pointer is *still installed* after you publish, because a node can be unlinked and retired in the instant between your read and your publish. The publish-then-verify protocol closes that window: if the re-read confirms `head == p`, then any retirement of `p` that happens *after* your verification will see your hazard pointer (because your store-with-fence happened-before the reclaimer's scan), and will defer. Any retirement that happened *before* your verification would have changed `head`, so your re-read would have failed and you would have restarted.

![timeline showing a thread publishing a hazard pointer to a node reading it safely and clearing it while the reclaimer skips the hazarded node](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-4.png)

When a thread removes a node, it does not free it. It puts the node on a per-thread **retired list**. Periodically — when the retired list crosses a threshold, say `R = 2 * (number of threads) * (hazard slots per thread)` — the thread runs a **scan**: it collects the union of all hazard pointers across all threads into a set, then walks its retired list and frees every node that is *not* in that set, keeping the rest for a later scan. The threshold guarantees amortized-constant work per retirement and bounds the number of un-freed nodes, which is the scheme's killer feature: **the number of nodes pinned in memory at any time is bounded by the number of hazard slots in use, which is `O(threads)`.** A stalled thread can pin at most as many nodes as it has hazard slots — one or two — not the whole structure.

### Hazard pointers in C++

Here is a real, idiomatic single-hazard-pointer implementation in C++ for the Treiber stack pop, with the bug fixed. C++26 standardizes `std::hazard_pointer`; here is a portable hand-rolled version that shows the mechanism explicitly:

```cpp
#include <atomic>
#include <vector>
#include <thread>

struct Node { int value; Node* next; };
std::atomic<Node*> head{nullptr};

constexpr int MAX_THREADS = 64;
// One hazard slot per thread. Single writer (the owner), many readers (scanners).
std::atomic<Node*> hazard[MAX_THREADS];
thread_local int tid = -1;  // assigned at thread start

// Each thread keeps its own retired list; freed lazily during scan.
thread_local std::vector<Node*> retired;

bool is_hazarded(Node* p) {
    for (int i = 0; i < MAX_THREADS; ++i)
        if (hazard[i].load(std::memory_order_acquire) == p) return true;
    return false;
}

void retire(Node* n) {
    retired.push_back(n);
    if (retired.size() < 2 * MAX_THREADS) return;   // batch: amortize the scan
    std::vector<Node*> keep;
    for (Node* p : retired) {
        if (is_hazarded(p)) keep.push_back(p);      // still in use: defer
        else delete p;                              // unreferenced: safe to free
    }
    retired.swap(keep);
}

bool pop(int& out) {
    Node* old;
    while (true) {
        old = head.load(std::memory_order_acquire);
        if (!old) return false;
        // Publish the hazard pointer, then VERIFY head is unchanged.
        hazard[tid].store(old, std::memory_order_seq_cst);  // store + full fence
        if (head.load(std::memory_order_acquire) != old)
            continue;                                // raced; republish and retry
        // old is now protected: any retirer will see our hazard pointer.
        Node* next = old->next;                      // SAFE dereference
        if (head.compare_exchange_weak(old, next,
                std::memory_order_acq_rel))
            break;
    }
    hazard[tid].store(nullptr, std::memory_order_release);   // clear: no longer using
    out = old->value;
    retire(old);                                     // do NOT delete; retire it
    return true;
}
```

The two changes that matter, against the broken version: the `hazard[tid].store(old, seq_cst)` followed by the re-read of `head` (the publish-then-verify), and `retire(old)` instead of `delete old`. The `seq_cst` store is the heavy hammer — it ensures the publication is globally visible before the subsequent verifying load, so a concurrent scanner cannot miss it. You can do this more cheaply with a `release` store plus a standalone `std::atomic_thread_fence(seq_cst)`, which is the usual optimization, but `seq_cst` on the store is the clearest correct version. The cost of that fence is the per-access price of hazard pointers, and we will measure it shortly.

### The bounded-memory guarantee

The property that makes hazard pointers the right choice when memory is precious: **a node is reclaimable as soon as every thread clears its hazard pointer to it, regardless of whether any thread is stalled.** A thread parked in the OS scheduler with no hazard pointer set pins nothing. A thread parked *while holding* a hazard pointer pins exactly the one node that slot points at. Across the whole system the un-freed set is bounded by the total number of hazard slots, which is `O(threads * slots-per-thread)`. This is why hazard pointers are used in places where a memory blowup is unacceptable — a long-running server that cannot afford a stalled reader to pin megabytes of retired nodes. The price you pay for that bound is the store-plus-fence on every protected access, and that price is real.

It is worth being precise about *why the fence is unavoidable*, because engineers reaching for `relaxed` orderings to speed this up reintroduce the bug. The publish at line (b) is a store to the hazard slot; the verify at line (c) is a load of `head`; and the reclaimer, on another core, does a load of the hazard slot followed by (logically) the free. For correctness we need the reclaimer to observe our hazard store *if* it has not yet decided to free — that is, our store must be ordered before our verifying load, and the reclaimer's load of our hazard slot must be ordered after it observed the head we verified against. On a machine with a store buffer (every modern CPU), a plain `release` store can sit in the store buffer, invisible to the reclaimer's core, while our own verifying load reads from cache and succeeds — so we conclude "protected" while the reclaimer, not yet seeing our store, concludes "unhazarded" and frees. The `seq_cst` store (or `release` store plus a `seq_cst` fence) drains the store buffer and establishes the total order that makes "I published before I verified" globally true, not just locally true. This is the [happens-before relation](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) doing load-bearing work: without the fence there is no happens-before edge from our publish to the reclaimer's scan, and the protocol's whole correctness argument evaporates. The fence is not a performance pessimization to be optimized away; it is the protocol.

There is also a liveness subtlety the re-read protocol resolves. Suppose two threads race to pop the same node, both publish a hazard pointer to it, and one wins the CAS. The loser's verify still saw `head == old` (it published before the winner's CAS landed), so the loser dereferences `old` safely — but `old` is being retired by the winner. That is fine: the winner's retire sees the loser's hazard pointer and defers; the loser finishes its (now-doomed) CAS, fails, clears its hazard, and retries. No double-free, no use-after-free, just a wasted iteration. The protocol is *lock-free*, not *wait-free* — a thread can be forced to retry an unbounded number of times under adversarial contention — but it always makes system-wide progress, which is the guarantee lock-free promises.

## Epoch-based reclamation: grace periods, cheaply

Hazard pointers make you pay on *every pointer you protect*. Epoch-based reclamation (EBR) makes a different bet: pay almost nothing per access, and instead reclaim in *batches* gated by a global notion of time called an epoch.

The mechanism: there is a single global epoch counter, an integer that only ever increases, plus a per-thread "local epoch" slot and a per-thread "active" flag. When a thread enters a critical section — about to touch the lock-free structure — it announces its participation by reading the global epoch and copying it into its local slot, and marking itself active. When it leaves, it marks itself inactive. A node removed from the structure is *retired into the current global epoch*: it is tagged with the epoch number `e` at which it was unlinked, and pushed onto a per-epoch limbo list.

The reclamation rule is the grace-period argument: **a node retired in epoch `e` may be freed once every thread has been observed in an epoch strictly greater than `e` (or inactive).** Why is that safe? Because a thread that is now in epoch `e+2` must have *re-entered* its critical section after the node was retired in `e` — and on re-entry it re-read the head and cannot be holding a pointer to a node that was already unlinked when it entered. Any thread that *was* holding the retired node entered during epoch `e` (or earlier) and was therefore observed in epoch `e`; for it to advance to `e+1` it must leave and re-enter, dropping its old pointer. So once nobody is observed in `e` or earlier, no live pointer to a node retired in `e` can exist. That is the grace period: the interval from "retired in `e`" to "everyone has advanced past `e`."

![before and after comparison showing an epoch retired node freed too early while a thread is still in the epoch versus freed safely after all threads advance](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-5.png)

To keep the bookkeeping `O(1)`, real EBR uses only **three epochs**: `e`, `e-1`, `e-2`, cycling modulo 3. To advance the global epoch from `e` to `e+1`, a thread checks that *every active thread is already in epoch `e`*; if so, it bumps the global counter, and the limbo list two epochs back (the one tagged `e-2`) is now safe to free wholesale, because at least two full epochs have elapsed since those nodes were retired. Three epochs are the minimum that lets you free one while another is being filled and another is the "draining" boundary.

The per-access cost is the headline win. Entering a critical section is a single relaxed-ish store of the global epoch into your local slot (plus a fence to order it before your reads). There is no per-pointer publication, no re-read-and-verify. You announce once at entry, do all the work, announce once at exit. For a traversal that touches a hundred nodes, hazard pointers pay a fence a hundred times; EBR pays once. That is the entire reason EBR exists.

#### Worked example: why two epochs of lag, not one

The "free after two epoch advances" rule looks arbitrary until you trace why one is not enough. Say the global epoch is 5, and thread T retires node N tagged epoch 5. Could we free N as soon as the global epoch reaches 6? No — consider thread U that entered its critical section while the epoch was 5, read a pointer to N (which was still linked at the moment U entered, because U entered before T unlinked it), and is still inside that section. For the global epoch to advance from 5 to 6, the advancing thread checked that every *active* thread was already in epoch 5 — and U *is* in epoch 5, so that check passes and the epoch becomes 6 while U is still holding N. Freeing N at epoch 6 would use-after-free U. Now advance again: for the epoch to reach 7, every active thread must be in epoch 6, which means U must have *left* epoch 5 and re-entered at 6 — and on re-entry U re-read the head and dropped its stale pointer to N. So by the time the epoch reaches 7 (two past N's retirement epoch of 5), no active thread can still hold N. That is the grace-period argument made concrete, and it is exactly why the limbo list freed when the epoch reaches `e+2` is the one tagged `e`, not `e+1`. The three-epoch ring is the minimum that lets one list drain while another fills and a third sits at the safe boundary.

### Epoch-based reclamation in Rust with crossbeam-epoch

The canonical, production-grade EBR implementation is the Rust crate `crossbeam-epoch`, which underpins `crossbeam`'s lock-free queues and is used throughout the Rust ecosystem. It wraps the protocol in an ergonomic API: you `pin()` the current thread (enter the critical section, getting a `Guard`), and you `defer_destroy` retired nodes through that guard. Here is a Treiber stack pop with EBR:

```rust
use crossbeam_epoch::{self as epoch, Atomic, Owned, Shared};
use std::sync::atomic::Ordering::{Acquire, Relaxed, Release};

struct Node<T> {
    value: T,
    next: Atomic<Node<T>>,
}

pub struct Stack<T> {
    head: Atomic<Node<T>>,
}

impl<T> Stack<T> {
    pub fn pop(&self) -> Option<T> {
        let guard = &epoch::pin();              // enter critical section: announce epoch
        loop {
            let head = self.head.load(Acquire, guard);   // protected by the guard
            match unsafe { head.as_ref() } {
                None => return None,            // empty
                Some(h) => {
                    let next = h.next.load(Acquire, guard);
                    // CAS head -> next
                    if self.head
                        .compare_exchange(head, next, Release, Relaxed, guard)
                        .is_ok()
                    {
                        unsafe {
                            // Do NOT free now. Defer until a grace period passes.
                            guard.defer_destroy(head);
                            return Some(std::ptr::read(&h.value));
                        }
                    }
                }
            }
        }                                       // guard drops here: leave critical section
    }
}
```

The `epoch::pin()` is the entire per-access cost: it announces the current epoch for this thread and returns a `Guard` that keeps the thread "in" the critical section for its lifetime. Every load you do `with` that guard is protected — you may dereference any pointer you read while the guard is live, because no node you can reach will be freed until your guard (and everyone else's that overlaps) has been dropped and the epoch has advanced twice. `guard.defer_destroy(head)` retires the node into the current epoch's limbo list; crossbeam frees it later, in a batch, once the grace period has elapsed. Notice there is no publish-then-verify dance and no per-pointer fence in the hot loop. That is EBR's bargain.

### The stalled-thread weakness

EBR's beautiful per-access cost comes with a sharp edge that you must respect: **a thread that enters a critical section and then stalls — gets descheduled, blocks on I/O, page-faults — pins its epoch, and therefore pins every node retired since, indefinitely.** Because the global epoch cannot advance while any active thread is stuck in an old epoch, the limbo lists grow without bound for as long as that thread is parked inside its critical section. One thread that calls `pin()` and then does something slow (or, worse, blocks) can make memory grow until the box runs out. This is the dual of hazard pointers' bounded-memory guarantee: HP bounds memory at the cost of per-access fences; EBR cheapens per-access at the cost of unbounded memory under a stall. Neither is universally better — it is a genuine trade, and the right answer depends on whether your critical sections are short and stall-free (EBR shines) or might block (hazard pointers are safer). The cardinal rule with EBR: **never block, never do I/O, never call anything slow while pinned.** Keep critical sections tiny and the weakness rarely bites; violate that and it bites hard.

## RCU: wait-free readers, deferred writer free

RCU — read-copy-update — is the scheme that powers the Linux kernel, and it takes EBR's "announce a region" idea to its logical extreme for the **read-mostly** case. The bet RCU makes is stark: **make readers as close to free as physically possible — wait-free, no atomics, no fences on the fast path — and push the entire cost of reclamation onto writers, who are assumed to be rare.**

![graph showing readers entering a read side critical section while a writer copies updates and publishes then defers the free until a grace period passes](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-6.png)

A reader brackets its access with `rcu_read_lock()` and `rcu_read_unlock()`. In the classic kernel implementation on a non-preemptible kernel, these are *nothing* — `rcu_read_lock()` is, astonishingly, a no-op or merely disables preemption; it emits no atomic instruction and no memory barrier. The reader then dereferences the shared pointer through `rcu_dereference()`, which is an ordinary load plus a dependency-ordering barrier (free on x86 and ARM, where data dependencies are respected) that ensures it sees a fully-initialized object. That is the entire read side. It does not write anything. It does not contend on any cache line. Multiple readers scale perfectly because they share nothing.

The writer does the work. To update, the writer **copies** the data it wants to change into a new node, **modifies** the copy, and then **publishes** the new node with a single store via `rcu_assign_pointer()`, which carries a release barrier so readers that pick up the new pointer see the fully-built object. After publishing, the old node is unlinked from the readers' perspective — new readers see the new node — but old readers may still be traversing the old node. So the writer cannot free the old node yet. It calls `synchronize_rcu()` (or registers a callback with `call_rcu()`), which **waits for a grace period**: an interval long enough that every reader that could have held a pointer to the old node has finished its read-side critical section. Once the grace period elapses, the old node is unreachable by any reader and the writer frees it.

The name is the recipe: **Read** (cheap, wait-free), **Copy** (the writer duplicates), **Update** (publish the copy, defer the free). The genius is the grace-period definition. How does the kernel know every reader is done without readers writing anything to check? Because `rcu_read_lock` disables preemption (in the classic flavor), a CPU that performs a **context switch** has provably exited any read-side critical section — you cannot be preempted inside a non-preemptible region. So a grace period is simply: *wait until every CPU has passed through a context switch (a quiescent state) since the update was published.* No reader bookkeeping at all; the scheduler's natural context switches are the liveness signal. That is why the read side can be free — the kernel infers reader completion from quiescent states it already tracks, not from anything readers do.

![matrix contrasting the RCU reader and writer on cost whether they block and what each one does](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-7.png)

### RCU in C, the kernel idiom

Here is the canonical RCU update of a linked-list node in the Linux kernel idiom, the read side and the write side side by side:

```c
struct config { int timeout; int retries; };
struct config __rcu *global_cfg;   // the shared pointer, RCU-annotated

// READER: wait-free, runs concurrently with the writer.
int read_timeout(void) {
    int t;
    rcu_read_lock();                            // enter: ~free, no atomics
    struct config *c = rcu_dereference(global_cfg);  // ordered load
    t = c->timeout;                             // safe: c stays valid in this section
    rcu_read_unlock();                          // leave
    return t;
}

// WRITER: copy, update, publish, then defer the free past a grace period.
void set_timeout(int new_timeout) {
    struct config *old, *new;
    new = kmalloc(sizeof(*new), GFP_KERNEL);
    old = rcu_dereference_protected(global_cfg, /* holds writer lock */ 1);
    *new = *old;                                // COPY
    new->timeout = new_timeout;                 // UPDATE the copy
    rcu_assign_pointer(global_cfg, new);        // PUBLISH (release barrier)
    synchronize_rcu();                          // wait for the grace period
    kfree(old);                                 // now safe: no reader holds old
}
```

Two writers must still serialize against each other with an ordinary lock — RCU coordinates readers-against-writers, not writers-against-writers. But readers never block and never spin; `read_timeout` is wait-free, which is why a path read millions of times per second (a routing table lookup, a config read, a `dentry` cache hit) is the perfect RCU customer. The `synchronize_rcu()` is the expensive part — it can take milliseconds, because it waits for a real grace period — which is exactly why RCU is for read-mostly data: you amortize one slow write against a flood of free reads. For non-blocking writers, `call_rcu(&old->rcu, free_callback)` registers a callback to free `old` after the grace period without blocking the writer, which is what hot kernel paths actually use.

### RCU in user space and the read-mostly bet

You do not need a kernel to use RCU. The `liburcu` (Userspace RCU) library provides the same primitives — `rcu_read_lock`, `rcu_dereference`, `synchronize_rcu`, `call_rcu` — in several flavors, including a "quiescent-state-based" flavor (QSBR) where threads periodically call `rcu_quiescent_state()` to signal they are between read-side sections, getting nearly the kernel's free read side. Rust has `crossbeam` and the `rcu` patterns built on epochs; C++ has folder-level RCU in `folly::rcu`. The shape is always the same: cheap wait-free readers, a writer that copies-publishes-defers.

The read-mostly bet is the whole story. RCU is *catastrophically* the wrong choice for write-heavy data: every write copies the object and pays a grace period, so a structure updated as often as it is read will drown in copies and grace-period waits. But for data read thousands of times per write — kernel data structures are overwhelmingly like this — RCU gives you reads that are not just lock-free but genuinely *free*, scaling linearly with cores because readers share no cache line. That asymmetry is why RCU is one of the most important concurrency mechanisms in systems software, and why it has its own section below.

## The trade-offs, head to head

Now we can lay the three schemes (plus GC) against each other on the three axes that actually decide which one you want: per-access cost, reclamation latency, and worst-case memory under a stall.

![matrix comparing hazard pointers epoch based reclamation RCU and garbage collection across per access cost reclamation latency and memory under a stall](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-2.png)

| Scheme | Per-access cost | Reclamation latency | Memory under a stall | Reader progress |
| --- | --- | --- | --- | --- |
| **Hazard pointers** | High: store + fence per protected pointer | Short: bounded retired-list scan | **Bounded**: `O(threads * slots)` | lock-free |
| **Epoch-based (EBR)** | Low: pin once per critical section | Medium: batched, gated on epoch advance | **Unbounded**: a stalled pinned thread pins everything | lock-free |
| **RCU** | Near zero: read side wait-free, no atomics | Long: a full grace period (ms in kernel) | **Unbounded**: a slow reader extends the grace period | **wait-free** |
| **Tracing GC** | Amortized: barriers + GC pauses | Nondeterministic: next GC cycle | Grows until next collection | depends on GC |

The three columns are in tension and you cannot win all three. Hazard pointers buy bounded memory by paying a fence on every access. EBR and RCU buy a cheap (or free) fast path by accepting that a stalled or slow thread can pin memory without bound. GC buys generality and zero programmer effort by accepting nondeterministic pauses and barrier overhead. **There is no scheme that is cheap per-access, immediate to reclaim, and bounded under a stall** — pick the two that matter for your workload and accept the third.

A second, subtler axis is **progress guarantee on the read side.** Hazard pointers and EBR keep reads lock-free (a thread always makes progress, but it may have to retry a CAS or a publish-verify). RCU makes reads *wait-free* — bounded steps, no retries, no atomics — which is a strictly stronger guarantee and the reason RCU reads scale so flatly with core count. If read latency tail matters (the p99.9 of a read must be tiny and predictable), RCU's wait-free read side is in a class of its own. If read latency is fine but you cannot tolerate a memory blowup, hazard pointers' bound is in a class of its own. EBR sits in the middle: cheaper than HP, bounded-er than RCU, and the easiest to drop in via crossbeam — which is why it is the pragmatic default for general-purpose lock-free Rust.

#### Worked example: choosing for a connection-tracking table

Concretely: you are building a connection-tracking table for a network proxy. Lookups happen on every packet — millions per second per core. Inserts and deletes happen on connection setup and teardown — orders of magnitude rarer. This is read-mostly by a factor of thousands. Walk the axes: per-access cost dominates because reads are the overwhelming majority, so you want the cheapest possible read; reclamation latency is nearly irrelevant because deletes are rare and a few stale entries cost nothing; memory under a stall matters only if readers can stall, and packet-processing readers are short and never block. That profile points unambiguously at **RCU** — wait-free reads that share no cache line, scaling linearly across the cores doing packet work, with the rare delete paying a grace period nobody on the read path notices. This is not a hypothetical; it is precisely why the Linux networking stack uses RCU for exactly these tables.

Now change one fact: the box is memory-constrained (an embedded router with 64 MB) and a reader *can* stall, because packet processing occasionally calls into a slow userspace helper. Suddenly RCU's unbounded-memory-under-a-stall is a liability — a stalled reader extends the grace period and retired entries pile up until you OOM. Now **hazard pointers** look better: bounded memory means a stalled reader pins at most its one or two hazard slots, never the whole table, and you trade some per-read cost for a hard memory ceiling you can actually fit in 64 MB. Same data structure, opposite scheme, because one constraint flipped. That is the entire discipline of choosing a reclamation scheme.

## How RCU powers the Linux kernel

RCU deserves its own section because it is not a niche technique — it is load-bearing infrastructure in the most widely deployed operating system kernel on earth, and seeing *where* it is used makes the read-mostly bet concrete.

The Linux kernel adopted RCU in 2002, with the core design and most of the implementation driven by Paul McKenney, who has written more about RCU than anyone alive. The kernel uses RCU in thousands of places, and they share a signature: a data structure read on a hot path far more often than it is written. The directory-entry cache (`dcache`) — looked up on every path resolution — uses RCU so that `stat`, `open`, and friends can walk the path with wait-free reads while renames and unlinks update under a lock and defer the free. The networking stack uses RCU for routing tables, the connection-tracking table, and the list of network devices — read per packet, written on configuration changes. The `IDR`/`XArray` ID allocators, the SELinux policy database, the list of loaded modules, the process credential structures — all RCU-protected, all read-mostly.

The payoff is measured in real scalability. Before RCU, these structures used reader-writer locks, and the reader lock — even though it allowed concurrent readers — still required every reader to *write* to the lock's cache line to take the read lock, which serialized cache-line ownership and capped read scalability at a handful of cores (the [readers-writer lock post](/blog/software-development/concurrency/readers-writer-locks-and-lock-granularity) explains exactly why a shared read lock still bottlenecks on the lock's cache line). RCU removed the reader-side write entirely: readers touch no shared lock cache line, so read throughput scales linearly to hundreds of cores. On a many-core server, swapping an `rwlock` for RCU on a read-mostly structure routinely turns a curve that flattens at 8 cores into one that climbs to 64 — not because the work changed, but because the readers stopped fighting over a cache line. That is the [cache-coherence](/blog/software-development/concurrency/cache-coherence-mesi-and-false-sharing) mechanism behind RCU's scalability: no shared write means no coherence traffic between readers, so reads are embarrassingly parallel. If there is one lesson from the kernel's RCU experience, it is that **the cost of a "cheap" reader lock is the cache line it writes**, and removing that write is worth a great deal on modern hardware.

The kernel's grace-period machinery is a marvel of engineering — it tracks quiescent states per CPU, batches grace periods so that thousands of `call_rcu` callbacks share one grace period, offloads callback processing to dedicated threads, and has special handling for CPUs that go idle or run in userspace (which are trivially quiescent). You do not need to reimplement any of that; you need to understand the bargain it strikes, which is the one we have been making all along: free reads, expensive writes, deferred frees gated by a grace period defined as "long enough that every prior reader has finished."

The batching is what makes the writer cost tolerable at scale, and it is worth understanding because it is the same trick EBR uses. A single grace period is expensive — milliseconds — but it is a *shared* resource: every node retired during one grace-period window is freed together when that window closes. So if a thousand updates land in the time it takes one grace period to elapse, they do not pay a thousand grace periods; they pay one, amortized across all thousand frees. This is why `call_rcu` (which registers a callback and returns immediately) is preferred over `synchronize_rcu` (which blocks) on hot paths: `call_rcu` lets the kernel batch your free with everyone else's into the next grace period, so the per-free cost approaches zero even though each grace period is individually slow. The rule to carry away is that a grace period is a *bus that leaves on a schedule*, not a taxi you hail per passenger — you pay for the bus once and everyone retired in that interval rides it. EBR's three-epoch batching is the userspace version of exactly this idea: retire cheaply, free in bulk when the boundary passes.

A subtle but important point for anyone tempted to use RCU outside the kernel: the kernel's "free" read side depends on `rcu_read_lock` disabling preemption, which is a privilege ordinary userspace code does not have. Userspace RCU therefore offers a spectrum of flavors that trade read-side cost for not needing kernel privileges — the "memory-barrier" flavor pays a fence per read (still cheaper than hazard pointers' per-pointer fence because it is per-section, not per-pointer), the "signal-based" flavor uses POSIX signals to detect quiescent states, and the QSBR flavor gets nearly the kernel's free read side but requires every reader thread to periodically call `rcu_quiescent_state()` to advance grace periods. There is no free lunch in userspace; you choose where on the read-cost-versus-intrusiveness curve you want to sit. When someone says "RCU has free reads," they mean the kernel's non-preemptible flavor specifically — repeat that claim about userspace QSBR and you are mostly right, about the memory-barrier flavor and you are wrong.

## Integrating a scheme with the Treiber stack and MS queue

Reclamation is not a standalone library you bolt on; it changes the shape of every structure that removes nodes. Let us be concrete about wiring it into the two canonical lock-free structures from the [building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures) post: the Treiber stack and the Michael-Scott (MS) queue.

For the **Treiber stack**, the only place a node leaves the structure is `pop`. So the integration is local: protect the head dereference in `pop` (publish a hazard pointer, or pin an epoch, or take the read lock), and retire the popped node instead of freeing it. `push` removes nothing and needs no reclamation work at all. The hazard-pointer `pop` above is exactly this; the crossbeam EBR `pop` is exactly this; they differ only in *which* protocol guards the dereference and how the retirement is expressed. One hazard slot suffices because `pop` holds exactly one node-pointer it must protect at a time.

The **MS queue** is more demanding because it removes from the head and also reads the tail and the head's `next` pointer, so a dequeue dereferences *two* nodes that another thread could free concurrently: the current dummy head and the node after it that becomes the new value-bearing head. A correct hazard-pointer dequeue therefore uses *two* hazard slots — one for `head`, one for `head->next` — each with its own publish-then-verify. Here is the structure of the protected MS-queue dequeue in C++ (eliding the full retry logic for clarity, showing the two-hazard protection):

```cpp
bool dequeue(int& out) {
    Node* h;
    Node* nxt;
    while (true) {
        h = head.load(std::memory_order_acquire);
        hazard[tid][0].store(h, std::memory_order_seq_cst);   // protect head
        if (head.load(std::memory_order_acquire) != h) continue;  // verify

        nxt = h->next.load(std::memory_order_acquire);
        hazard[tid][1].store(nxt, std::memory_order_seq_cst); // protect next
        if (head.load(std::memory_order_acquire) != h) continue;  // re-verify head

        if (nxt == nullptr) return false;                     // empty
        out = nxt->value;
        if (head.compare_exchange_weak(h, nxt,
                std::memory_order_acq_rel))
            break;                                            // dequeued
    }
    hazard[tid][0].store(nullptr, std::memory_order_release);
    hazard[tid][1].store(nullptr, std::memory_order_release);
    retire(h);                                                // old dummy: retire
    return true;
}
```

The pattern generalizes: **count the maximum number of node-pointers a single operation dereferences simultaneously, and that is how many hazard slots that operation needs.** A stack pop: one. An MS-queue dequeue: two. A lock-free linked-list traversal with a "previous" and "current" cursor: two. A skip list traversal at height `h`: up to `h`. This is also why hazard pointers can get awkward for deeply pointer-chasing structures — you need a hazard slot per simultaneously-held pointer, and the scan cost grows with the total slot count. For those structures, EBR's "pin once, dereference anything" model is far more ergonomic, which is another reason crossbeam-epoch is the default for complex Rust lock-free structures while hazard pointers dominate where the pointer-holding is shallow and bounded memory is the priority.

## A complete worked example: a hazard-pointer-protected pop

Let us put the whole hazard-pointer protocol together as one runnable, end-to-end example you can read top to bottom, with the correctness argument annotated inline. This is the worked example the post promises: reclaiming a popped node safely with hazard pointers.

#### Worked example: a safe pop, step by step with the invariant at each line

```cpp
#include <atomic>
#include <vector>
#include <cassert>

struct Node { int value; Node* next; };

class HPStack {
    std::atomic<Node*> head_{nullptr};
    static constexpr int kThreads = 64;
    std::atomic<Node*> hp_[kThreads]{};            // one hazard slot per thread
    static thread_local std::vector<Node*> retired_;

    bool hazarded(Node* p) {
        for (int i = 0; i < kThreads; ++i)
            if (hp_[i].load(std::memory_order_acquire) == p) return true;
        return false;
    }
    // The batch scan is inlined into pop() below (step h): collect all
    // hazard pointers, free every retired node named by none, keep the rest.
public:
    void push(int v) {                              // push removes nothing: no HP needed
        Node* n = new Node{v, nullptr};
        n->next = head_.load(std::memory_order_relaxed);
        while (!head_.compare_exchange_weak(n->next, n,
                std::memory_order_release, std::memory_order_relaxed)) {}
    }

    bool pop(int tid, int& out) {
        Node* old;
        while (true) {
            old = head_.load(std::memory_order_acquire);     // (a) read head
            if (!old) return false;                          //     empty
            hp_[tid].store(old, std::memory_order_seq_cst);  // (b) PUBLISH hazard
            if (head_.load(std::memory_order_acquire) != old)// (c) VERIFY
                continue;                                    //     raced: republish
            // INVARIANT after (c): hp_[tid]==old AND head_==old were both true,
            // so any retirer of old that runs after (b) sees our hazard and defers.
            Node* next = old->next;                          // (d) SAFE deref
            if (head_.compare_exchange_weak(old, next,
                    std::memory_order_acq_rel))              // (e) unlink
                break;
        }
        hp_[tid].store(nullptr, std::memory_order_release);  // (f) clear hazard
        out = old->value;
        retired_.push_back(old);                             // (g) RETIRE, do not free
        if (retired_.size() >= 2 * kThreads) {               // (h) batch scan
            std::vector<Node*> keep;
            for (Node* p : retired_) {
                if (hazarded(p)) keep.push_back(p);           //     still in use
                else delete p;                                //     safe to free
            }
            retired_.swap(keep);
        }
        return true;
    }
};
thread_local std::vector<Node*> HPStack::retired_;
```

Trace the correctness. The use-after-free in the naive version happened because between reading `old` and dereferencing `old->next`, another thread freed `old`. Here, line (b) publishes `old` into our hazard slot with a full fence, and line (c) re-reads `head_` and confirms it still equals `old`. If those both hold, then at the instant of line (c), `old` was the head *and* our hazard pointer named it. Any thread that retires `old` from this point forward runs `hazarded(old)` in its scan (line h equivalent), sees our hazard pointer, and keeps `old` on its retired list instead of freeing it. So when we dereference `old->next` at line (d), `old` is guaranteed alive. After we unlink and finish, line (f) clears our hazard, and line (g) retires `old` rather than freeing it — because some *other* thread might still be in its own window (a)–(c) holding `old`. Only a scan that finds `old` in no hazard slot frees it (line h). Every freed node was, at the moment of freeing, named by no hazard pointer; that is the invariant, and it is exactly the guarantee a garbage collector would have given us.

Note `push` takes no hazard pointer: it adds a node and never dereferences a node another thread could free (the node it pushes is brand new and private until the CAS publishes it). The reclamation machinery touches only the removal path. That is the general rule — **protect dereferences of shared, removable nodes; ignore allocations and pure structural adds.**

## Measured behavior: overhead and reclamation latency, honestly

Numbers make the trade real, but they must be honest. The figures below are *representative orders of magnitude* from published benchmarks and the kind of microbenchmarks you would run yourself on a typical modern x86 server (single socket, dozens of cores, TSO memory model); they are not measurements from one canonical run, and the real numbers swing with allocator, core count, contention, NUMA topology, and the memory model of your hardware. Treat them as "what shape should I expect" and then **measure your own workload** — warm up the caches, run for seconds not milliseconds, run many times, and report the distribution, not one number.

The per-access cost first. Hazard pointers pay a store plus a sequentially-consistent fence on every protected pointer. On x86 a `seq_cst` store compiles to a locked instruction or an `mfence`, which costs on the order of tens of nanoseconds and, crucially, drains the store buffer and serializes — so a tight loop that protects one pointer per iteration sees a real, measurable slowdown versus an unprotected loop. Order of magnitude: a hazard-pointer-protected access can be roughly 2x–5x the cost of the bare atomic load it protects, dominated by the fence. EBR's `pin()` amortizes across the whole critical section: enter once, touch many nodes, so per-node cost approaches the bare load — often within 10–20% of unprotected for a multi-node traversal. RCU's read side on a non-preemptible kernel emits *no* atomic and *no* fence beyond a dependency barrier that is free on x86 and ARM, so a protected read is essentially indistinguishable from an unprotected one — the read side is, to a first approximation, free.

| Scheme | Per-access read overhead (representative) | What dominates the cost |
| --- | --- | --- |
| No scheme (unsafe) | baseline (and wrong) | the bare atomic load |
| **RCU read side** | ~0%, near baseline | nothing on the fast path; dependency barrier only |
| **EBR (crossbeam)** | ~10–20% over baseline, amortized | one pin per critical section, spread over many derefs |
| **Hazard pointers** | ~2x–5x the bare load, per pointer | the seq_cst store + fence on every protected pointer |
| **Tracing GC** | varies; barrier + pause cost | write barriers plus stop-the-world or concurrent GC pauses |

Now reclamation latency and memory. Hazard pointers reclaim eagerly — a scan runs whenever the retired list crosses its threshold, and the bound on un-freed nodes is `O(threads * slots)`, so on a 64-thread workload with one slot each you pin at most ~64 nodes beyond the batching threshold. That is a hard, small ceiling. EBR reclaims in epoch batches: a node retired in epoch `e` waits until the global epoch reaches `e+2`, so reclamation lag is "two epoch advances," which is fast when threads cycle through critical sections quickly and *infinite* when one thread pins an epoch by stalling. RCU's reclamation lag is a full grace period — in the Linux kernel, a grace period is typically on the order of *milliseconds*, because it waits for every CPU to pass through a quiescent state, and `synchronize_rcu()` blocks the writer for that long (which is why hot paths use `call_rcu` to defer without blocking). So RCU's memory lag is the largest of the three on the happy path, but readers never pay for it.

#### Worked example: measuring the stall pathology

The most important thing to measure is not steady-state throughput — it is the failure mode. Here is the experiment that exposes EBR's and RCU's Achilles' heel and hazard pointers' strength. Run a producer-consumer workload on a lock-free queue with reclamation, and at a random point, **pause one consumer thread for 500 ms inside its critical section** (simulating a descheduled thread or a page fault). Watch resident memory.

Under hazard pointers, memory barely moves: the paused thread pins exactly the one or two nodes its hazard slots name; every other retired node is freed on schedule. Under EBR or RCU, memory *climbs the entire 500 ms*, because the paused thread pins its epoch (EBR) or extends the grace period (RCU), so nothing retired during the pause can be freed; you watch resident size grow linearly with the pause, then snap back down when the thread resumes and the epoch finally advances. On a high-throughput queue retiring a million nodes a second, a 500 ms stall under EBR pins half a million nodes — tens or hundreds of megabytes — that hazard pointers would never have pinned. This is the single most decision-relevant measurement in the whole post: **if your threads can stall while inside a lock-free operation, and you cannot afford the memory, hazard pointers' bound is not a nicety, it is the requirement.** Measure it on your own structure before you choose; the steady-state throughput numbers will not warn you, only the stall experiment will.

A final honesty note on the memory model. Every number above assumes x86 TSO, where a `seq_cst` store is "merely" a store-buffer drain. On a weakly-ordered architecture — ARM, POWER — the fences hazard pointers and the publish barriers RCU needs are *more* expensive relative to plain loads, because the hardware does less ordering for free, so the relative cost of hazard pointers' per-access fence is higher on ARM than on x86. If you benchmark on an x86 laptop and deploy on ARM servers, your reclamation overhead can be meaningfully larger in production than in your test. Name your platform when you report; the answer genuinely differs.

## Case studies / real-world

**Maged Michael's hazard pointers (2004).** The scheme was introduced in Maged Michael's paper *"Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"* (IEEE Transactions on Parallel and Distributed Systems, 2004), building on his earlier work on lock-free queues. The paper's contribution was not just the protocol but the *proof* that it gives bounded memory and is itself lock-free, and the recognition that the per-access fence is the price of that bound. Hazard pointers were patented (by IBM, where Michael worked), which slowed their adoption in open source for years and is part of why EBR became popular as an unencumbered alternative; the patents have since expired, and hazard pointers are now standardized in C++26 as `std::hazard_pointer`, with Facebook's `folly::hazptr` a widely used production implementation. The lesson for an engineer: when you need a hard memory bound under adversarial scheduling, this is the scheme with a *proof* of that bound, and it is now first-class in the standard library.

**Linux kernel RCU (2002–present), Paul McKenney.** RCU entered the Linux kernel in 2002 and is documented exhaustively by Paul McKenney, including his dissertation *"Exploiting Deferred Destruction"* and the kernel's own `Documentation/RCU` tree. The scale is staggering: RCU is invoked in thousands of call sites and is credited with much of the kernel's read-side scalability on many-core machines. McKenney's repeated public point is that RCU's value is *empirical and measured* — it was adopted because it turned flat read-scaling curves (capped by reader-writer-lock cache-line contention) into linear ones, not because it was elegant in the abstract. The case study to remember: the directory-entry cache (`dcache`) path-walk was converted to RCU and the read side became wait-free, which is why a modern Linux box can resolve paths concurrently across hundreds of cores without the readers fighting over a lock. RCU is the proof that "make the reader free and the writer pay" is the right bet for read-mostly kernel data.

**crossbeam-epoch (Rust ecosystem).** The `crossbeam-epoch` crate, descended from Aaron Turon's work and now maintained as part of the `crossbeam` project, is the de facto EBR implementation for Rust and the reclamation engine under `crossbeam`'s lock-free queues and deques (which in turn power work-stealing schedulers like the one in `rayon` and historically in `tokio`). It demonstrates the EBR bargain in production: a `pin()`-based API that makes the per-access cost a single epoch announcement and lets you `defer_destroy` retired nodes, with the three-epoch protocol freeing in batches. The crate's documentation is candid about the stall weakness — it explicitly warns not to hold a `Guard` across a blocking call — which is exactly the failure mode we measured. crossbeam-epoch is the reason a Rust engineer can write a correct lock-free structure without ever writing reclamation code by hand; it is also the reason "don't block while pinned" is a piece of Rust folk wisdom.

## When to reach for this (and when not to)

A reclamation scheme is overhead and complexity you take on only because you have no choice. The decision tree below is the short version, and the rest of this section walks each branch; the intuition is to start from the two cheap escape hatches (no sharing, or a GC) before committing to any manual protocol at all.

![matrix mapping read mostly write heavy bounded memory and GC available workloads to the reclamation scheme that fits and the reason](/imgs/blogs/memory-reclamation-hazard-pointers-epochs-and-rcu-8.png)

The matrix above collapses the whole choice into four rows because, in practice, one constraint usually dominates. If a tracing collector is available, you are done — none of the manual schemes apply. If the workload is read-mostly, RCU's free reads win so decisively that nothing else is close. If memory must be bounded under any scheduling, hazard pointers' `O(threads)` cap is the requirement. And if writes are frequent, the per-write copy cost of RCU rules it out and you fall back to checking the writer cost of epoch-based or hazard pointers directly. Read the rows top to bottom and stop at the first one your workload satisfies; that is your scheme.

**Reach for a manual scheme only if you have no GC and you genuinely need a lock-free structure.** The first question is always: do you have a tracing garbage collector that sees this memory? If yes — Java, Go, C#, ordinary heap objects — then *use it* and write zero reclamation code. The JVM's `ConcurrentLinkedQueue` and Go's lock-free patterns simply drop references; the GC is your reclamation scheme and it is correct and fast. The single most common mistake in this whole area is hand-rolling hazard pointers in a language whose GC would have done it for free. Only reach for a manual scheme in C, C++, or unsafe Rust, or when managing native/off-heap memory the GC does not track.

**The second question: is a lock-free structure even justified?** Reclamation is the tax on going lock-free, and lock-free is only worth it when a lock is provably your bottleneck — measured contention, a profile showing the lock at the top, a many-core box where the lock serializes everything (the [building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures) post makes this case). If a plain `Mutex` or a `RWMutex` is not actually hurting you, a lock-free structure plus a reclamation scheme is a large complexity cost for no benefit. Measure first; do not go lock-free, and therefore do not take on reclamation, on a hunch.

Once you are committed to a GC-free lock-free structure, choose by the workload:

- **Read-mostly, reads on a hot path, writers rare → RCU.** Routing tables, config, caches read far more than written. You get wait-free reads that scale linearly across cores and share no cache line; the rare writer pays the grace period. This is RCU's home turf and nothing else comes close on the read side. Do *not* use RCU for write-heavy data — the per-write copy and grace period will crush you.
- **Bounded memory is a hard requirement, or threads can stall mid-operation → hazard pointers.** When a stalled reader pinning unbounded memory would OOM you (embedded, tight memory budgets, adversarial scheduling), the `O(threads)` bound is the requirement, not a preference. Accept the per-access fence as the price of the bound. Also prefer hazard pointers when each operation holds only one or two pointers at once (stacks, simple queues) — they get awkward when you must hold many pointers deep in a structure.
- **General-purpose lock-free in Rust, short stall-free critical sections, cheap reads wanted → epoch-based (crossbeam).** EBR is the pragmatic middle and the easiest to adopt: pin once, dereference freely, defer destruction, let the crate do the rest. The hard rule is *never block, do I/O, or call anything slow while pinned* — keep critical sections tiny and EBR's unbounded-under-stall weakness rarely fires. For deeply pointer-chasing structures (skip lists, trees) EBR's "pin once, deref anything" ergonomics beat hazard pointers' slot-per-pointer accounting.

**When not to reach for any of this:** if your data is not actually shared across threads (thread-local, or owned by one actor — see [the actor model](/blog/software-development/concurrency/the-actor-model-mailboxes-isolation-and-supervision)), there is no reclamation problem because there is no concurrent reader to race the free. If your access pattern is fine under a lock, stay under the lock. If you are in a GC language, let the GC work. Reclamation schemes are a sharp tool for a narrow, real problem — lock-free structures in manual-memory languages — and a liability everywhere else.

## Key takeaways

1. **The hard part of lock-free is not the CAS — it is deciding when `free` is safe.** A lock gives you the lifetime boundary for free; remove the lock and you must reconstruct, by an explicit protocol, the moment at which no thread can still hold a pointer to a retired node.
2. **Use-after-free is the real bug, and the ABA problem is its symptom.** A node freed while another thread holds a pointer into it causes a segfault or a silent CAS-success on reused memory. Reclamation is the cure for both.
3. **In a GC language, the problem vanishes — let the GC work.** The single most common mistake is hand-rolling hazard pointers where a tracing collector would have reclaimed correctly and fast. Reach for a manual scheme only in C/C++/unsafe-Rust or for native memory.
4. **Hazard pointers buy bounded memory with a per-access fence.** Publish the pointer you are about to use, re-verify the head, defer the free until no hazard pointer names the node. Memory pinned is `O(threads * slots)` even under a stall — the killer feature when memory is precious.
5. **Epoch-based reclamation buys a cheap fast path by risking unbounded memory under a stall.** Announce an epoch once per critical section, retire into the current epoch, free a node only after every thread has advanced two epochs past its retirement. Never block while pinned.
6. **RCU makes readers wait-free and free, and pushes all cost onto rare writers.** Read-copy-update: cheap bracketed reads, writer copies-publishes-defers, free after a grace period. The right and only choice for read-mostly hot paths; catastrophic for write-heavy data.
7. **The three axes are in tension: per-access cost, reclamation latency, memory under a stall — pick two.** No scheme wins all three. RCU and EBR cheapen access by risking memory; hazard pointers bound memory by paying per access; GC trades pauses for zero effort.
8. **The number of hazard slots an operation needs equals the pointers it dereferences at once.** A stack pop needs one; an MS-queue dequeue needs two; a deep traversal needs many — which is when EBR's "pin once" ergonomics win.
9. **Measure the stall, not just the throughput.** Steady-state numbers hide the failure mode; pause a thread inside a critical section and watch resident memory to see EBR/RCU pin unboundedly and hazard pointers hold the line. And name your platform — x86 TSO and ARM weak ordering change the per-access cost.

## Further reading

- **Maged M. Michael, "Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"** (IEEE TPDS, 2004) — the original paper; the protocol, the proof of bounded memory, and the lock-free property. The source for everything in the hazard-pointer sections here.
- **Paul E. McKenney, "Is Parallel Programming Hard, And, If So, What Can You Do About It?"** (the "perfbook," freely available) and the Linux kernel `Documentation/RCU/` tree — the definitive treatment of RCU, grace periods, quiescent states, and the kernel's use of them.
- **Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming*** — the textbook on lock-free progress, linearizability, and the foundations these schemes rest on; chapters on concurrent data structures put reclamation in context.
- **Anthony Williams, *C++ Concurrency in Action* (2nd ed.)** — the practical C++ angle on lock-free structures, hazard pointers, and the memory orderings the protocols depend on.
- **The `crossbeam-epoch` crate documentation** (docs.rs) — a clear, honest description of EBR in practice, including the explicit warnings about holding a `Guard` across a blocking call.
- **The series intro and capstone** — [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) and [the concurrency playbook for choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model), for where reclamation sits in the larger arc of mechanisms, alongside the sibling on [building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures) that these schemes complete.
- **For the hardware story underneath all of this — why a shared cache line is expensive and why RCU's no-shared-write reads scale** — see [the memory hierarchy from registers to HBM](/blog/machine-learning/high-performance-computing/the-memory-hierarchy-registers-shared-memory-and-hbm).
