---
title: "Compare-and-Swap and Building Lock-Free Data Structures"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Turn a single compare-and-swap instruction into a Treiber stack, a Michael-Scott queue, and an honest answer to when lock-free is worth it."
tags:
  [
    "concurrency",
    "parallelism",
    "lock-free",
    "compare-and-swap",
    "treiber-stack",
    "michael-scott-queue",
    "cas",
    "data-structures",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-1.png"
---

A mutex is a promise: "while I hold this, nobody else touches the shared thing." It works, and most of the time it is the right answer. But there is a class of problem where that promise becomes the bottleneck. Picture sixteen threads all pushing onto one shared stack a few million times a second. With a mutex, fifteen of them are always asleep, queued behind the one holding the lock, and the structure spends most of its life as a single-lane road with a traffic jam. The CPU has sixteen cores; the lock has one lane.

Lock-free data structures take a different promise. There is no "while I hold this." Instead, every mutation is staged privately and then **published in a single atomic instruction** — one `compare-and-swap` — that either lands or doesn't. If it doesn't land, it's because some other thread published first, so you re-read the current state and try again. No thread ever blocks waiting on another thread's critical section. The structure is always in a consistent state, between every pair of instructions, because the only thing that ever changes it is that one indivisible publish. That is the whole idea, and the rest of this post is about turning it into working code.

We will build two real structures from `compare-and-swap`: the **Treiber stack** (lock-free push and pop by swapping the head pointer) and the **Michael-Scott queue** (a lock-free FIFO with a separate head and tail, and a clever "help advance the tail" trick that keeps any thread from stalling the whole queue). We will hit the famous **ABA problem** — where a node gets popped, freed, recycled, and a stale CAS is fooled into thinking nothing changed — and fix it with tagged pointers and hazard pointers. And we will measure honestly: lock-free wins big under heavy contention, but it can *lose* at low contention, and the genuinely hard part is not the algorithm at all — it's knowing when you are allowed to free a node. The figure below traces the heartbeat of every one of these structures: a CAS loop that fails once because a rival won the race, then re-reads and succeeds.

![timeline of a compare and swap loop that reads the head, builds a node, fails the first swap because a rival won, re-reads, and succeeds on the second swap](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-1.png)

If you have not yet read why concurrency forces these choices on us at all, the series intro [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) sets the frame: shared mutable state plus nondeterministic scheduling is the hazard, and every technique in this series is a different way to tame it. Lock-free is the most aggressive tame — it removes the lock entirely — and it is also the easiest to get subtly, catastrophically wrong.

## What "lock-free" actually promises (and what it doesn't)

Before we write a line of code, it is worth being precise about the word, because "lock-free" is thrown around loosely and the precise meaning is what makes the algorithms correct.

A data structure is **lock-free** if, whenever threads run for long enough, *at least one* of them makes progress — completes its operation — in a bounded number of its own steps, regardless of what the others do, even if some of them are suspended mid-operation by the OS scheduler. The key clause is "even if some are suspended." With a mutex, if the thread holding the lock is descheduled (it ran out of its time slice, or it page-faulted, or the kernel preempted it to run something else), *every other thread waiting on that lock is stuck* until it gets rescheduled. The whole system's progress is hostage to one thread. Lock-free forbids that: no single thread's stall can freeze the structure for everyone.

Note the careful "at least one." Lock-free does **not** promise that *every* thread finishes — a specific unlucky thread could lose the CAS race over and over and retry forever in theory. That stronger guarantee is **wait-free**: every thread finishes in a bounded number of its own steps, no exceptions, no starvation. Wait-free is harder to build and usually slower in the common case, and most production "lock-free" structures (including the two we build here) are lock-free but not wait-free. The full ladder — obstruction-free, lock-free, wait-free, and what each one buys you — is its own topic, covered in [the progress hierarchy: blocking, lock-free, and wait-free](/blog/software-development/concurrency/the-progress-hierarchy-blocking-lock-free-and-wait-free). For now hold one fact: lock-free means *the structure as a whole always moves*, even though an individual thread might have to retry.

The mechanism that buys this is `compare-and-swap`. Let me state it precisely, because every algorithm below is just this one operation in a loop.

### The compare-and-swap primitive, exactly

`compare-and-swap(addr, expected, new)` does three things as one indivisible step:

1. Read the value at `addr`.
2. If it equals `expected`, store `new` at `addr` and report success.
3. Otherwise leave `addr` unchanged and report failure (usually returning the value it actually found).

The whole thing is atomic. No other thread can observe `addr` between the read and the conditional store; no other thread can sneak a write in between. On x86 this compiles to a single `lock cmpxchg` instruction. On ARM and other weakly-ordered machines it is built from a `load-linked / store-conditional` (LL/SC) pair, where the store conditionally fails if anything touched the line since the load — same effect, different hardware path. The mechanics of how that single instruction is implemented, and how it is the building block for a mutex too, are in [how a lock is built: test-and-set, CAS, and spinlocks](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks). Here we use CAS not to *build* a lock but to *avoid* one.

In C++ the primitive is `std::atomic<T>::compare_exchange_strong` (or `_weak`); in Java it is `AtomicReference.compareAndSet`; in Rust it is `AtomicPtr::compare_exchange`; in Go it is `atomic.CompareAndSwapPointer`. They all have the same shape. The C++ signature is worth memorizing because it exposes a detail the others hide:

```cpp
// expected is passed by reference: on failure it is OVERWRITTEN with the
// value actually found, so your retry loop already has the fresh value.
std::atomic<Node*> head;
Node* expected = head.load(std::memory_order_relaxed);
bool ok = head.compare_exchange_weak(
    expected,                  // what I think head is
    new_node,                  // what I want head to become
    std::memory_order_release, // success ordering
    std::memory_order_relaxed  // failure ordering
);
// if ok == false, `expected` now holds the real current head.
```

Two ordering arguments matter and we will not gloss over them. The **success** order must at least be `release` on a publish, so that everything you wrote into the node before the CAS is visible to a thread that later reads the node with `acquire`. The **failure** order can be `relaxed` because on failure you publish nothing — you just retry. If you are fuzzy on why `release` on the publisher pairs with `acquire` on the reader to create a happens-before edge, [atomics and memory orderings: from relaxed to seq-cst](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst) is the prerequisite. The short version: without that pairing, a reader could see the new `head` pointer but *not* the node's fields, and dereference garbage.

## The CAS loop idiom

Almost all lock-free code is the same three-line dance, repeated until it succeeds. Read the current value. Compute the new value from it. Try to atomically swap the old for the new. If the swap fails, *someone else changed the value between your read and your swap*, so your computed-new is stale — throw it away, re-read, recompute, and try again. The first figure of this post is exactly that loop with one retry; in code it is a `while` loop whose body is "read, compute, attempt to publish," and whose exit condition is "the publish landed."

Here is the idiom stripped to its bones — an atomic increment built by hand, so you can see the loop without any data-structure noise:

```cpp
// Atomic increment via CAS, just to show the loop shape.
// (In real code you'd use fetch_add; this is the CAS skeleton.)
void increment(std::atomic<int>& x) {
    int cur = x.load(std::memory_order_relaxed);
    while (!x.compare_exchange_weak(
               cur, cur + 1,
               std::memory_order_relaxed)) {
        // CAS failed: `cur` was just overwritten with the real value.
        // Loop body re-runs with the fresh `cur`. No re-load needed.
    }
}
```

Read it carefully. `cur` is our snapshot of `x`. We try to swap `cur` for `cur + 1`. If another thread incremented `x` in the meantime, `x` no longer equals our `cur`, the CAS fails, and — this is the ergonomic trick of `compare_exchange` — `cur` is *automatically updated* to the value actually found. So the loop body re-runs with a fresh snapshot, recomputes `cur + 1` from it, and tries again. No update is ever lost, because the only write to `x` is the CAS, and the CAS only writes if `x` still holds the value we based our computation on.

This is the same pattern that fixes the classic lost-update race. If you wrote `x = x + 1` with a plain load, an add, and a plain store, two threads could both read the same old value, both add one, and both store the same new value — one increment vanishes. That interleaving is dissected in [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition). The CAS loop closes the gap: the read-modify-write is no longer three separate steps that can be interleaved; the *commit* is one atomic step that detects interference and retries.

### `weak` vs `strong`, and the spurious-failure footgun

Notice I used `compare_exchange_weak`. The weak form is allowed to fail *spuriously* — to report failure even when `addr` did equal `expected` — because on LL/SC machines the store-conditional can fail for benign reasons (an interrupt, an unrelated write to the same cache line). The strong form retries internally to hide spurious failures. The rule of thumb:

- In a **loop** (which CAS almost always is), use `weak`. A spurious failure just means one extra harmless iteration, and `weak` compiles to tighter code on ARM.
- For a **single, non-looping** CAS, use `strong`, so a spurious failure doesn't make you incorrectly conclude the value changed.

#### Worked example: two threads, one counter, the interleaving

Let me make the "no lost update" claim concrete with an exact interleaving. Suppose `x = 5`, and threads T1 and T2 each run the CAS-loop `increment`.

| step | T1 | T2 | `x` in memory |
| --- | --- | --- | --- |
| 1 | `cur = load() = 5` | | 5 |
| 2 | | `cur = load() = 5` | 5 |
| 3 | | CAS(5 to 6) **succeeds** | 6 |
| 4 | CAS(5 to 6) **fails**, `cur` becomes 6 | | 6 |
| 5 | CAS(6 to 7) **succeeds** | | 7 |

Both increments survive: `x` ends at 7, not 6. The lost-update interleaving is impossible because T1's CAS at step 4 checks `x == 5`, finds `x == 6`, fails, and re-reads. Compare this to a plain `x = x + 1`: at step 4 T1 would blindly store `6`, clobbering T2's increment, and `x` would wrongly end at 6. The CAS *detects* the conflict that the plain store ignores. Every lock-free structure below is this same detect-and-retry, just with a pointer instead of an integer.

## The Treiber stack: lock-free push and pop

A stack is the simplest place to see CAS build a real structure, which is exactly why R. Kent Treiber used it in his 1986 IBM report. A stack is a singly-linked list where you only ever touch one end: the **head**. Push prepends a node; pop removes the head node. Both operations change exactly one word — the head pointer — and that is what makes the stack a perfect CAS target.

The figure contrasts the locked version with the lock-free one: instead of taking a mutex around the head update, we CAS the head directly, holding no lock at any point.

![before and after comparison of a stack protected by a mutex around push versus a Treiber stack that swaps the head with a single compare and swap](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-2.png)

### Push, in C++

```cpp
struct Node {
    int value;
    Node* next;
};

std::atomic<Node*> head{nullptr};

void push(int v) {
    Node* node = new Node{v, nullptr};
    Node* old_head = head.load(std::memory_order_relaxed);
    do {
        node->next = old_head;            // link new node to current head
    } while (!head.compare_exchange_weak( // publish: head := node
                 old_head, node,
                 std::memory_order_release,
                 std::memory_order_relaxed));
}
```

Trace it. We allocate the node and read the current head into `old_head`. We point `node->next` at `old_head` — the new node now sits in front of the old top, *but nothing is published yet*; the stack still points at `old_head`. Then the CAS tries to swing `head` from `old_head` to `node`. If it succeeds, in one atomic step the head now points at our node, whose `next` is the old head — the stack grew by one, consistently. If it fails, another thread pushed (or popped) in the meantime, so `old_head` was overwritten with the new real head; we relink `node->next` to *that* and retry. The `release` order guarantees that `node->value` and `node->next`, written before the CAS, are visible to any thread that later loads `head` with `acquire` and walks into our node.

The graph below shows the two paths through a push: the straight line to "published" on a successful CAS, and the branch to a retry node when a rival wins. Note it is acyclic — a retry is a *branch to a fresh attempt*, not a loop back in time.

![graph of a Treiber stack push showing the node linked to the old head then a compare and swap that either publishes or branches to a retry node that re-reads and tries again](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-3.png)

### Pop, in C++

```cpp
bool pop(int& out) {
    Node* old_head = head.load(std::memory_order_acquire);
    while (old_head != nullptr) {
        Node* next = old_head->next;      // read the second node
        if (head.compare_exchange_weak(   // publish: head := next
                old_head, next,
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            out = old_head->value;
            delete old_head;              // <-- DANGER: is this safe? (see ABA)
            return true;
        }
        // CAS failed: old_head refreshed; loop re-reads next.
    }
    return false; // stack empty
}
```

Pop reads the head, reads the head's `next`, and CASes the head from `old_head` to `next`. On success the head node is unlinked and we return its value. Same loop, same retry-on-conflict. But look at the `delete old_head` line — I flagged it `DANGER` deliberately. That `delete` is the single most dangerous line in lock-free programming, and it is the doorway to the ABA problem and to the whole memory-reclamation question. We will earn the right to that `delete` later; for now, just notice that the *algorithm* is trivially correct but *freeing memory* is not.

### The same stack in Rust and Java

The idioms differ enough across languages to be worth showing. Rust's borrow checker hates raw shared mutation, so a lock-free stack uses `AtomicPtr` and `unsafe`, paired with an explicit memory order:

```rust
use std::sync::atomic::{AtomicPtr, Ordering};
use std::ptr;

struct Node {
    value: i32,
    next: *mut Node,
}

pub struct Stack {
    head: AtomicPtr<Node>,
}

impl Stack {
    pub fn push(&self, value: i32) {
        let node = Box::into_raw(Box::new(Node { value, next: ptr::null_mut() }));
        let mut old = self.head.load(Ordering::Relaxed);
        loop {
            unsafe { (*node).next = old; }                 // link
            match self.head.compare_exchange_weak(         // publish
                old, node, Ordering::Release, Ordering::Relaxed,
            ) {
                Ok(_) => break,
                Err(actual) => old = actual,               // retry with fresh head
            }
        }
    }
}
```

Java hides the raw pointers behind references and gives you `AtomicReference` with a clean `compareAndSet`:

```java
import java.util.concurrent.atomic.AtomicReference;

class TreiberStack<E> {
    private static final class Node<E> {
        final E value;
        Node<E> next;
        Node(E value) { this.value = value; }
    }

    private final AtomicReference<Node<E>> head = new AtomicReference<>();

    void push(E value) {
        Node<E> node = new Node<>(value);
        Node<E> oldHead;
        do {
            oldHead = head.get();      // read current head
            node.next = oldHead;       // link
        } while (!head.compareAndSet(oldHead, node)); // publish, retry on fail
    }

    E pop() {
        Node<E> oldHead, next;
        do {
            oldHead = head.get();
            if (oldHead == null) return null;
            next = oldHead.next;
        } while (!head.compareAndSet(oldHead, next)); // publish
        return oldHead.value;
    }
}
```

The Java version, crucially, has *no ABA problem and no reclamation problem* — and that is not because Java's algorithm is cleverer. It is because the garbage collector will not free a node while any thread still holds a reference to it. The hardest part of the C++ and Rust versions — when is it safe to `delete`/`free` — is *solved for you by the GC* in Java. Hold that thought; it is the single biggest practical reason lock-free is easier on a managed runtime.

Go sits between the two camps: it has a garbage collector (so no manual `free`, like Java) but exposes raw `unsafe.Pointer` CAS primitives in `sync/atomic` (so the code looks closer to C++). A Treiber push in Go reads almost identically — the same read-link-CAS-retry loop — with the GC quietly handling the reclamation that C++ would agonize over:

```go
package stack

import (
	"sync/atomic"
	"unsafe"
)

type node struct {
	value int
	next  unsafe.Pointer // *node
}

type Stack struct {
	head unsafe.Pointer // *node
}

func (s *Stack) Push(v int) {
	n := &node{value: v}
	for {
		old := atomic.LoadPointer(&s.head) // read current head
		n.next = old                       // link (private until published)
		if atomic.CompareAndSwapPointer(&s.head, old, unsafe.Pointer(n)) {
			return // published in one CAS
		}
		// CAS failed: someone else pushed/popped; loop re-reads.
	}
}
```

Four languages, one algorithm. The *concept* — read the head, link privately, publish with one CAS, retry on conflict — is identical across C++, Rust, Java, and Go. What differs is exactly two things: the spelling of the atomic primitive, and *who owns the reclamation problem*. In Java and Go the GC owns it; in C++ and Rust you do. That single difference is the entire reason the same algorithm is an afternoon's work in one language and a multi-week, expert-reviewed effort in another.

## Why a single atomic publish keeps the structure consistent

Here is the load-bearing invariant of every lock-free structure, stated plainly: **between any two instructions, the structure is in a valid state, because the only operation that ever mutates it is one atomic CAS that transitions it from one valid state directly to another.** There is no intermediate, half-updated, observable state — no moment where the head points "halfway" into a node, no window where a reader sees a torn pointer.

Contrast this with a non-atomic multi-word update. Inserting into a doubly-linked list lock-free would require updating the new node's `prev` and `next`, *and* the neighbors' pointers — four writes. There is no single instruction that does all four atomically, so between them the list is malformed, and a concurrent reader can crash. That is precisely why lock-free doubly-linked lists are notoriously hard and rarely worth it, and why the structures that *work* well lock-free are the ones whose every operation reduces to changing **one** word. The Treiber stack works because push and pop each change exactly the head pointer. The Michael-Scott queue works because enqueue and dequeue each reduce to one or two carefully-ordered single-word CASes, each leaving the queue valid.

This is the design heuristic to carry forward: **find the one word whose atomic swap publishes your whole change, and stage everything else privately before you touch it.** In the Treiber push, the private staging is "allocate the node and set `node->next`"; the publish is the head CAS. The node is invisible to everyone until the instant the CAS lands. If the CAS fails, the node is *still* private — no one ever saw it — so retrying is free and safe. The mechanism that makes "invisible until published" hold across cores is the `release`/`acquire` memory ordering; without it the *pointer* could be published before the *fields*, and another core could follow the pointer into uninitialized memory. The publish is atomic *and* ordered, and both properties are required.

### Linearizability: the formal version of "looks atomic"

There is a precise word for the property we just described informally: **linearizability**. An operation is linearizable if it appears to take effect *instantaneously* at some single moment between its invocation and its return — its **linearization point**. For the Treiber push, the linearization point is *the successful head CAS*: before that instant the node is not in the stack; after it, it is; there is no moment where it is "partially" in. Every concurrent execution of linearizable operations is equivalent to *some* sequential ordering that respects real-time precedence. This is the gold standard of correctness for concurrent data structures, and it is exactly what the single-atomic-publish design buys you for free: because the only state change is one atomic instruction, *that instruction is the linearization point*, and the operation trivially appears atomic.

This is also why the *two*-CAS Michael-Scott enqueue is still linearizable even though it is not a single instruction: its linearization point is the **first** CAS — the one that links the node into `tail->next`. The instant that CAS lands, the node is reachable by walking from the head, so it is *in the queue*, observable by any dequeue. The second CAS (swinging the tail) changes no observable queue contents — it only updates a hint — so it is *not* a linearization point. That is the deep reason the algorithm can split enqueue across two instructions and still behave atomically: only one of them is the moment of truth, and the other is recoverable bookkeeping any thread can finish. When you design a lock-free structure, the first question to answer is "what is the linearization point of each operation?" If you cannot name a single instant where each operation takes effect, the structure is probably not linearizable and probably not correct.

The connection to the memory model is direct: the linearization point must also be a **happens-before** edge. The successful CAS, done with `release`, *synchronizes-with* a later `acquire` load of the same location by another thread, establishing that everything the publisher did before the CAS is visible to the reader after the load. Linearizability is the *what* (operations look instantaneous and ordered); the release/acquire happens-before relation is the *how* (the hardware and compiler actually make the writes visible in that order). The formal definitions of happens-before and synchronizes-with are in [memory models: sequential consistency and happens-before](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before); for lock-free code the practical consequence is that every publish must carry at least `release` and every read that follows a published pointer must carry at least `acquire`, or the "instantaneous" illusion shatters and readers see torn or stale state.

### The torn-read sibling problem

One more reason the publish must be a single word: a pointer-sized value must be read and written *atomically* by the hardware, or a reader can observe a half-old, half-new pointer — a **torn read**. On a 64-bit machine an aligned 8-byte pointer load/store is atomic, so a plain `head.load()` is safe. But the moment you try to atomically swap *more* than a pointer — say a pointer *plus* a version tag, 16 bytes — you need double-width CAS or you risk tearing. Torn reads, and their cousin the time-of-check-to-time-of-use bug, are covered in depth in [the ABA problem, TOCTOU, and torn reads](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads); they are the same family of hazard, all rooted in "the value changed between when I looked and when I acted."

It is worth being concrete about what "atomic for the hardware" means, because it is a property of the *type and alignment*, not of your intent. An `int` is atomic if it is naturally aligned (its address is a multiple of its size) and fits a single machine word; a misaligned or oversized value can straddle two cache lines, and a store to it becomes two separate writes that another core can catch half-done. This is why `std::atomic<T>` may insert a hidden lock for large or oddly-sized `T` (check `is_lock_free()` — if it returns false, the "atomic" is secretly a mutex, and you have lost the lock-free property without noticing). A pointer is the sweet spot: exactly one word, naturally aligned, genuinely lock-free on every modern ISA. That is the deeper reason the well-behaved lock-free structures publish through a *pointer* swap specifically — it is the largest thing the hardware will swap atomically without help. Step up to a pointer-plus-counter pair and you are at the 16-byte boundary where `cmpxchg16b` (x86-64) or a paired LL/SC (ARM) is required, and where some platforms simply cannot do it, forcing the pointer-packing fallback.

## The Michael-Scott queue: a lock-free FIFO

A stack is easy because both ends are the same end. A FIFO queue is harder: you enqueue at the **tail** and dequeue at the **head**, and the two ends move independently. Maged Michael and Michael Scott published the canonical lock-free queue in 1996, and it has been the standard ever since — Java's `ConcurrentLinkedQueue` is a direct descendant. The algorithm is a small masterpiece, so we will go slowly.

The queue is a singly-linked list with two atomic pointers, `head` and `tail`, and a permanent **dummy node** at the front. The dummy is the trick that makes the empty-queue and one-element cases stop being special: `head` always points at the dummy, and the first *real* element is `dummy->next`. An empty queue has `head == tail == dummy` and `dummy->next == nullptr`.

The figure traces an enqueue: read the tail, CAS the new node into `tail->next`, then swing the tail forward — and if the tail is lagging, *any* thread that notices can help advance it.

![timeline of a Michael-Scott queue enqueue reading the tail, linking the new node with a compare and swap, then advancing the tail, with another thread helping when the tail lags](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-4.png)

### Enqueue: two CASes, and "helping"

The genius of the algorithm is that enqueue is split into **two** atomic steps, and the structure is valid after *each* one:

1. **Link the node**: CAS `tail->next` from `nullptr` to the new node. After this, the node is *in* the queue (reachable from head by walking `next`), but `tail` still points at the old last node — the tail "lags" by one.
2. **Swing the tail**: CAS `tail` from the old last node to the new node. This just catches `tail` up to reality.

Why split it? Because there is no single instruction that links the node *and* moves the tail atomically. So we make the *link* the real publish — once `tail->next` points at the node, the node is enqueued, period — and treat moving the tail as bookkeeping that can be finished *by anyone*. This is the **helping** principle: if thread A links a node but gets descheduled before it swings the tail, thread B comes along, *sees* that `tail->next` is non-null (the tail is lagging), and swings the tail forward *on A's behalf* before doing its own work. No thread ever waits for A to wake up. That is what makes the queue lock-free: A's stall cannot freeze the queue.

```cpp
struct Node { int value; std::atomic<Node*> next; };
std::atomic<Node*> head, tail;   // both start at a dummy node

void enqueue(int v) {
    Node* node = new Node{v, nullptr};
    Node* t;
    while (true) {
        t = tail.load(std::memory_order_acquire);
        Node* next = t->next.load(std::memory_order_acquire);
        if (t == tail.load(std::memory_order_acquire)) { // tail still consistent?
            if (next == nullptr) {
                // tail is the real last node: try to link our node.
                if (t->next.compare_exchange_weak(
                        next, node,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                    // linked! now try to swing the tail (best-effort).
                    tail.compare_exchange_strong(
                        t, node,
                        std::memory_order_release,
                        std::memory_order_relaxed);
                    return;
                }
            } else {
                // tail is lagging: HELP advance it, then retry.
                tail.compare_exchange_strong(
                    t, next,
                    std::memory_order_release,
                    std::memory_order_relaxed);
            }
        }
    }
}
```

Read the `else` branch carefully — that is the helping. If `next != nullptr`, it means some other thread already linked a node onto `tail->next` but hasn't swung `tail` yet. Rather than wait, *we* swing the tail forward (`CAS tail from t to next`) and loop to retry our own enqueue against the now-correct tail. The tail-swing CASes can fail harmlessly: if two threads both try to advance the tail, only one wins, the other's CAS fails, and the tail still ends up correct. Failure of the *bookkeeping* CAS is never an error — it just means someone else did the bookkeeping for you.

### Dequeue

```cpp
bool dequeue(int& out) {
    Node *h, *t, *next;
    while (true) {
        h = head.load(std::memory_order_acquire);
        t = tail.load(std::memory_order_acquire);
        next = h->next.load(std::memory_order_acquire);
        if (h == head.load(std::memory_order_acquire)) {
            if (h == t) {                 // head and tail coincide
                if (next == nullptr) return false; // queue empty
                // tail lagging: help advance it before we dequeue.
                tail.compare_exchange_strong(t, next,
                    std::memory_order_release, std::memory_order_relaxed);
            } else {
                out = next->value;        // read value from the next node
                if (head.compare_exchange_weak( // publish: head := next
                        h, next,
                        std::memory_order_release,
                        std::memory_order_relaxed)) {
                    // retire(h);  // <-- reclamation hazard, see below
                    return true;
                }
            }
        }
    }
}
```

Dequeue advances `head` past the dummy: the value comes from `head->next` (the first real node), and after the CAS, that node *becomes* the new dummy. The old dummy is unlinked and — there it is again — needs to be freed, which is the reclamation hazard. Notice dequeue *also* helps advance a lagging tail (the `h == t` and `next != nullptr` case): even a reader pitches in to keep the writer's bookkeeping current. Helping flows both directions.

The matrix below catalogs the standard lock-free structures and the one trick each one leans on — the stack's single-CAS publish, the queue's tail-helping, the hashmap's per-bucket CAS, and the ring buffer's fetch-add slot claim.

![matrix listing the Treiber stack, Michael-Scott queue, lock-free hashmap, and ring buffer against the hot word each one swaps and the trick that keeps it consistent](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-5.png)

### A word on the lock-free hashmap and the ring buffer

The matrix mentions two more structures worth a sentence each. A **lock-free hash map** (Cliff Click's is the famous one; split-ordered lists by Shalev and Shavit are the textbook design) CASes individual buckets, so writers to different keys never contend — the "one hot word" is per-bucket, not global. A **bounded ring buffer** (the heart of the LMAX Disruptor) often does not even need CAS for the common case: producers claim a slot with an atomic **fetch-add** on a sequence counter, which *never fails* and so never retries — a much cheaper primitive than CAS when the operation is a commutative counter bump. The lesson, again: the structure's performance comes from choosing the cheapest atomic that publishes its change. Fetch-add beats CAS when you can use it, because fetch-add has no retry loop at all.

## The ABA problem returns

Now the snake in the grass. Go back to the Treiber `pop` and its `delete old_head`. Consider this sequence on a stack with nodes A on top of B on top of C:

#### Worked example: ABA fools a Treiber pop

1. **Thread 1** starts `pop`. It reads `head` and gets A. It reads `A->next` and gets B. It is *about to* CAS `head` from A to B — and right here the OS deschedules T1. T1 is frozen holding `old_head = A`, `next = B`.
2. **Thread 2** runs. It pops A (CAS head A to B, succeeds), and **frees A**. It pops B (CAS head B to C, succeeds), frees B. The stack is now just C. So far so good.
3. **Thread 2** pushes a new value, and the allocator — being a good allocator — hands back the *just-freed address of A* for the new node. Call it A' but it lives at the same address as A. T2 pushes it: `head` now points at A' (same bits as A), and `A'->next = C`.
4. **Thread 1 wakes up.** Its CAS is "swap `head` from A to B". `head` currently holds A' — *the same bits as A*. **The CAS succeeds.** It sets `head = B`.

Disaster. B was freed in step 2. `head` now points at deallocated memory. The next pop dereferences a dangling pointer. T1's CAS was fooled because the head's *value* (the address bits) was A at the start and A again at the end — even though the head had changed to B, to C, and back to A-the-address in between. CAS compares values, and the value came back the same. **A, then B, then A: ABA.**

The before/after figure shows the trap and the fix side by side: a plain CAS on the bare address is fooled when A is recycled, while a version-tagged pointer makes the recycled A compare unequal because the version moved on.

![before and after comparison showing a plain compare and swap fooled when address A is freed and reused versus a tagged pointer whose version counter makes the reused address compare unequal](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-6.png)

The root cause is the same as every hazard in [the ABA problem, TOCTOU, and torn reads](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads): you checked a condition (head == A), the world changed, and then you acted on the stale check (CAS using A). CAS *almost* solves time-of-check-to-time-of-use because it rechecks at the moment of action — but it rechecks the *value*, and ABA is precisely the case where the value returns to its old self while its meaning has changed underneath. Pop is vulnerable; the simple `increment` earlier was *not*, because an integer counter that returns to 5 genuinely is back at 5 — there is no aliasing of identity. ABA bites when the value is a **reused identity** (a recycled pointer), not a plain number.

## The fixes: tagged pointers, DWCAS, and hazard pointers

There are two families of fix, and they attack the problem from opposite ends. Tagged pointers make the value *not* come back the same. Hazard pointers make sure the memory is *not* reused while anyone might still CAS against it.

### Fix 1 — tagged (versioned) pointers + double-width CAS

Attach a monotonically increasing **version counter** to the pointer, and CAS the pair `(pointer, version)` together. Every time you publish, you bump the version. Now even if the *pointer* returns to A, the *version* has moved on, so `(A, 7)` and `(A, 9)` compare unequal and the stale CAS fails — exactly as it should.

The catch is that you must CAS *two* words atomically: the pointer and the counter. That requires **double-width CAS** (DWCAS) — `cmpxchg16b` on x86-64, which atomically compares and swaps a 128-bit value. C++ exposes it through `std::atomic` on a 16-byte struct (with `-mcx16`); on platforms without it you fall back to **pointer packing**, stealing the high bits of a pointer for a smaller version tag (virtual addresses don't use all 64 bits, so you have ~16 bits to spare — enough that wraparound is rare in practice but, honestly, not *impossible*).

```cpp
struct TaggedHead {
    Node* ptr;
    uint64_t tag;            // version counter, bumped on every publish
};

std::atomic<TaggedHead> head; // needs 16-byte atomic (cmpxchg16b)

bool pop(int& out) {
    TaggedHead old = head.load(std::memory_order_acquire);
    while (old.ptr != nullptr) {
        TaggedHead next{old.ptr->next, old.tag + 1}; // bump the version
        if (head.compare_exchange_weak(
                old, next,
                std::memory_order_acquire,
                std::memory_order_relaxed)) {
            out = old.ptr->value;
            // still must defer the free safely; tag stops ABA, not use-after-free
            return true;
        }
    }
    return false;
}
```

Tagged pointers stop the *logical* ABA — the CAS no longer succeeds wrongly. But read the comment: they do **not** by themselves solve use-after-free. If T1 reads `old.ptr->next` while T2 has already freed `old.ptr`, you crash *before* the CAS even runs. The tag stops the CAS from succeeding, but the dereference already happened on dead memory. So tagged pointers are necessary for correctness but not sufficient for safe reclamation. That distinction trips up a lot of people: the version counter defends the *swap*, not the *load*.

Java sidesteps the DWCAS plumbing with `AtomicStampedReference`, which pairs a reference with an `int` stamp and CASes both:

```java
import java.util.concurrent.atomic.AtomicStampedReference;

AtomicStampedReference<Node> head =
    new AtomicStampedReference<>(null, 0);

boolean compareAndPublish(Node oldHead, Node newHead, int stamp) {
    return head.compareAndSet(oldHead, newHead, stamp, stamp + 1);
}
```

But again, on the JVM the GC already prevents the use-after-free, so the stamp is only for the rare logical-ABA case where the *same live object* legitimately leaves and re-enters the structure. In unmanaged C++/Rust, the version counter and the reclamation problem are separate fights you must win both of.

### Fix 2 — hazard pointers (the reclamation fix)

The deeper fix attacks reuse directly: **don't free a node while any thread might still be about to access it.** A hazard pointer is a single-writer, multi-reader slot, one per thread, in which a thread *announces* "I am currently looking at this node — do not free it." Before a thread dereferences a node it loaded from a shared pointer, it writes that node's address into its hazard slot and *re-validates* that the shared pointer still points there. When a thread wants to free a node, it does not `free` immediately; it puts the node on a private **retire list**, and only actually frees nodes whose addresses appear in *no* thread's hazard slot. The protocol guarantees: if I announced a node as hazardous before you scanned the slots, you will see my announcement and defer the free; if I announced after your scan, then the pointer I loaded must already be stale and I'll re-validate and back off.

```cpp
// Sketch of the hazard-pointer protocol for Treiber pop.
Node* protect(std::atomic<Node*>& src, HazardSlot& my_hp) {
    Node* p;
    do {
        p = src.load(std::memory_order_acquire);
        my_hp.store(p, std::memory_order_release); // announce intent
    } while (p != src.load(std::memory_order_acquire)); // re-validate
    return p; // now safe to dereference: nobody will free p
}

void retire(Node* n, RetireList& rl) {
    rl.push(n);
    if (rl.size() >= kThreshold) {
        auto hazards = scan_all_hazard_slots();   // collect every thread's HP
        for (Node* candidate : rl.drain())
            if (!hazards.contains(candidate)) delete candidate; // safe free
            else rl.keep(candidate);               // still hazardous, try later
    }
}
```

Hazard pointers turn the unbounded "when can I free?" question into a bounded scan: a node can be freed once no hazard slot names it. The cost is a memory fence per protected load and a periodic scan, but it gives **safe, bounded-memory reclamation without a GC**. This is the production answer for C++ lock-free structures, and it is now standardized as `std::hazard_pointer` in C++26. The full design space — hazard pointers vs epoch-based reclamation vs RCU, and the trade-offs between them — is its own deep topic, covered in [memory reclamation: hazard pointers, epochs, and RCU](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu). For this post, the point is: **the algorithm is the easy 20%; safe reclamation is the hard 80%.**

The matrix below lines up the atomic building blocks — load, CAS, DWCAS, fetch-add — with what each one is for in lock-free code and the gotcha that comes with it.

![matrix mapping atomic load, compare and swap, double-width compare and swap, and fetch-add to their role in lock-free code and the main gotcha of each](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-8.png)

## Helping and the lock-free guarantee

We met helping in the queue's `else` branch; now let me name why it is not just an optimization but the *thing that makes the structure lock-free*. The lock-free guarantee is "the structure always makes progress even if a thread stalls mid-operation." In the Michael-Scott queue, the dangerous mid-operation stall is exactly between linking a node and swinging the tail. If only the original thread were allowed to swing the tail, then a thread descheduled right after linking would leave the tail permanently lagging until it woke up — and every other enqueue would be blocked behind it. That would be a *blocking* structure wearing a lock-free costume.

Helping removes the dependency. Because *any* thread that observes a lagging tail will advance it, the original thread's stall is irrelevant: the next thread to touch the queue finishes the half-done operation and proceeds. No thread's progress depends on any specific other thread waking up. That is the lock-free property, made real. The same idea recurs across lock-free algorithms — a thread that finds another's operation half-applied either completes it or rolls it back so it can proceed — and it is the conceptual price of admission for stronger progress guarantees. Wait-free algorithms generalize helping further: every operation announces itself in a shared descriptor, and any thread can complete *anyone's* announced operation, which is how you guarantee that *no* thread ever starves.

A subtle and important consequence: helping means the *same* logical operation may be completed by a *different* thread than the one that started it, and the CAS that "would have" been done by the original thread now fails (because someone helped). That failed CAS is, again, **not an error** — it is the signal "your work is already done." Designing lock-free code is largely about making every CAS failure mean something benign and recoverable: either "someone published before me, re-read and retry" or "someone finished my bookkeeping, carry on." If a CAS failure can ever mean "the structure is now corrupt," the algorithm is wrong.

Here is the precise reason the tail-swing CAS is safe to fail. The CAS is `compare_exchange(tail, t, node)` — it only succeeds if `tail` *still* equals `t`, the old tail this thread observed. If two threads both try to advance the tail from `t` to `node`, the first wins and the second's CAS finds `tail == node` (not `t`), fails, and does nothing — but the tail is *already* where it should be, so there is nothing left to do. If, instead, the tail had already been advanced *past* `node` by some later operation, this thread's CAS also fails (`tail != t`), and again that is correct: the tail is already ahead, helping is unnecessary, move on. In *every* outcome of the bookkeeping CAS — win, lose-to-a-peer, lose-to-the-future — the tail ends up correct and no thread is misled. That property, that a failed CAS leaves the structure in a state where the failure is *self-evidently fine*, is what you are really verifying when you prove a lock-free algorithm correct. It is not enough that the happy path works; you must check that every CAS failure is a benign no-op or a clean retry, because under contention those failures are the *common* case, not the exception.

## The limits: memory reclamation is the actually hard part

Let me be blunt about where the difficulty lives, because the literature makes lock-free look like it is about clever CAS sequences and it is not. The CAS sequences are the *easy* part. You can read the Treiber and Michael-Scott papers in an afternoon and write the pointer-juggling correctly. The part that takes a career to get right — the part that ships bugs into production — is **memory reclamation**: in a language without a garbage collector, *when is it safe to free a node?*

The problem has no easy answer because lock-free means you cannot take a lock to coordinate the free, and a node you want to free might be in the hand of a thread that read its address a microsecond ago and is *about to* dereference it. Every reclamation scheme is a different bargain:

- **Leak it.** Never free. Correct, trivial, and occasionally the right answer for short-lived programs or arena-allocated pools — but obviously a non-starter for a long-running service.
- **Reference counting.** Atomically increment a per-node refcount before use, decrement after, free at zero. Correct but slow — every access touches a shared counter, which reintroduces contention on the very cache line you were trying to avoid, and naive atomic refcounting has its own ABA on the count.
- **Hazard pointers.** Announce-and-scan, as above. Bounded memory, ~constant overhead per access, the modern default. The cost is a fence per protected load.
- **Epoch-based reclamation (EBR).** Threads pass through global "epochs"; a node freed in epoch N is reclaimed once every thread has advanced past N. Lower per-access cost than hazard pointers (no per-load fence), but a single stalled thread can stall reclamation and grow memory unbounded — the classic EBR failure mode.
- **RCU (read-copy-update).** The Linux kernel's workhorse: readers are nearly free (no atomics on the read side), writers wait for a grace period during which all pre-existing readers finish. Phenomenal for read-mostly data, but writers pay and readers must be brief.

There is no free lunch. Managed runtimes (Java, Go, C#) hand you the GC, which is *exactly* a sophisticated reclamation scheme, and that is why lock-free is dramatically more pleasant on the JVM — the hardest 80% is already solved. In C++ and Rust you own the 80%. Rust's ownership model helps with *some* of it (you can't accidentally use a freed node if you model lifetimes right), but shared lock-free structures still reach for `unsafe` and a crate like `crossbeam-epoch`, which implements EBR for you. The honest summary: **if you are writing lock-free in a non-GC language and you have not thought hard about reclamation, your code is wrong, even if it passes every test** — because the ABA-and-use-after-free window is narrow and your tests probably don't hit it.

## Stress-testing the design: what breaks at the edges

A lock-free structure that works in a two-thread unit test can fail in production in ways that only appear under specific, adversarial conditions. Let me walk the edges deliberately, because reasoning about them is the actual engineering skill — the algorithm is the easy part, as we keep saying.

**What happens at 64 threads on a 64-core box?** The Treiber stack has a hidden scaling wall that the contention table didn't show: *every* push and pop CASes the *same* head pointer, so all 64 threads are hammering one cache line. Under MESI, that line ping-pongs between cores' caches — each successful CAS invalidates the other 63 copies, and the losers re-fetch the line across the interconnect before retrying. This is a contention *hotspot*, and it means the Treiber stack, while lock-free, does not *scale* indefinitely — past some core count, retry rates climb and throughput flattens or even dips. The cache-line ping-pong cost is the same mechanism behind false sharing, detailed in [cache coherence, MESI, and false sharing](/blog/software-development/concurrency/cache-coherence-mesi-and-false-sharing). The fix at very high core counts is often *not* a better lock-free stack but a *different structure* — per-core sub-stacks with occasional rebalancing (an elimination-backoff stack), or simply accepting that a single LIFO hotspot is inherently unscalable and rearchitecting so threads don't all share one stack. Lock-free removes the *blocking*, not the *coherence traffic*.

**What happens on ARM's weak memory model versus x86?** On x86 (TSO), loads and stores are barely reordered, so `acquire`/`release` compile to plain loads and stores with almost no extra cost — the hardware already gives you most of the ordering. Port the *exact same code* to ARM or POWER, whose memory models reorder aggressively, and those `acquire`/`release` annotations now emit real barrier instructions (`ldar`/`stlr` on ARM). Two consequences: the code is *slower* on ARM relative to its uncontended baseline (real barriers cost), and — more dangerously — code that was *accidentally correct* on x86 because TSO hid a missing barrier will *break* on ARM. This is a classic porting bug: a lock-free structure that passed every test on an x86 CI machine corrupts data on an ARM server. The lesson: never use `relaxed` ordering on a publish "because it worked on my x86 laptop"; the memory model is part of the algorithm's correctness, not a performance knob.

**What happens when the retry loop never wins?** Lock-free guarantees *some* thread progresses, not *every* thread. A pathologically unlucky thread on a hot stack can lose the CAS race repeatedly while others keep winning — it makes no progress while the structure as a whole does. In practice this is rare and self-correcting (the winners eventually back off), but under sustained extreme contention it manifests as a long-tail latency spike for the unlucky thread. The mitigation is **exponential backoff**: after a failed CAS, spin or sleep for a randomized, growing interval before retrying, which spreads out the contenders and lets stragglers through. If your latency SLO is on the *tail* (p99.9), an unbounded retry loop is a liability, and you may want a wait-free structure or a different design. This starvation-under-contention failure mode is the same family as the issues in [livelock, starvation, and priority inversion](/blog/software-development/concurrency/livelock-starvation-and-priority-inversion).

**What happens when the allocator is the bottleneck?** If push `new`s and pop `delete`s on every operation, you may be measuring `malloc`'s internal lock, not your stack. Worse, frequent free-then-reuse is exactly what *arms* the ABA gun — fast recycling makes address reuse common. A node **pool** (free-list of pre-allocated nodes) avoids both: it removes allocator contention *and* lets you control reuse timing so the reclamation scheme stays in charge. But a pool reintroduces its own lock-free free-list (a Treiber stack of free nodes!), with its own ABA exposure on the pool's head — so you have not escaped the problem, you have moved it. This recursion — every lock-free structure tends to need another lock-free structure underneath for memory — is part of why the field is deep.

The thread you should pull on first when a lock-free structure misbehaves: it is almost never the CAS sequence (that's textbook and likely right) — it is the reclamation, the memory ordering, or the contention pattern. Race detectors and stress harnesses are your friends here; see [finding concurrency bugs: race detectors and stress testing](/blog/software-development/concurrency/finding-concurrency-bugs-race-detectors-and-stress-testing) for how to actually catch these, since a normal test run almost never will.

## Progress guarantees compared

Before the line-by-line walk, one comparison table to anchor where these structures sit on the progress ladder — because "lock-free" is one rung, and choosing the rung is a real design decision.

| Property | Blocking (mutex) | Obstruction-free | Lock-free | Wait-free |
| --- | --- | --- | --- | --- |
| A stalled thread can freeze others? | yes | no | no | no |
| At least one thread always progresses? | no (holder can stall all) | only if run in isolation | yes | yes |
| *Every* thread finishes in bounded steps? | no | no | no | yes |
| Starvation of one thread possible? | yes | yes | yes | no |
| Typical cost in the common case | lowest when uncontended | low | low to moderate | highest |
| Our structures | — | — | Treiber, Michael-Scott | rare specialized designs |

Read it as a price ladder: each stronger guarantee costs more in the common case. The Treiber stack and Michael-Scott queue are **lock-free** — a stalled thread can't freeze the structure, but a specific thread *can* starve under adversarial contention. Wait-free buys you no-starvation at a real cost, and is reserved for cases where bounded per-thread latency is a hard requirement (some real-time systems). The full ladder, with the consensus-number theory behind why some objects can build wait-free structures and others can't, is in [the progress hierarchy: blocking, lock-free, and wait-free](/blog/software-development/concurrency/the-progress-hierarchy-blocking-lock-free-and-wait-free).

## Worked example: a lock-free stack push, line by line

Let me put it all together and walk a single Treiber push instruction by instruction, with the ABA hazard called out exactly where it would bite a careless pop running concurrently.

#### Worked example: pushing 42 onto a stack of B on C

State at the start: `head -> B -> C -> null`. Thread P wants to push `42`.

```cpp
void push(int v) {                    // v = 42
    Node* node = new Node{v, nullptr};// 1. allocate node N at some address
    Node* old_head =                  // 2. read head: old_head = B
        head.load(std::memory_order_relaxed);
    do {
        node->next = old_head;        // 3. link N->next = B  (PRIVATE; nobody sees N yet)
    } while (!head.compare_exchange_weak(
                 old_head, node,      // 4. publish: CAS head from B to N
                 std::memory_order_release,
                 std::memory_order_relaxed));
}
```

- **Line 1** allocates node N. N is *private* — it exists only in thread P's hand; no shared pointer reaches it. Nothing concurrent can see N yet.
- **Line 2** snapshots the head: `old_head = B`. This is the time-of-check. From here until the CAS, the head *could* change under us.
- **Line 3** links N to B. The stack is *still* `head -> B -> C`; N points into the stack but the stack does not point at N. State is consistent; N is invisible.
- **Line 4** is the publish: CAS `head` from B to N. Two outcomes.
  - **Success**: in one atomic step, `head -> N -> B -> C`. The push is done, consistently, with no intermediate observable state. The `release` order ensures N's `value=42` and `next=B`, written in lines 1 and 3, are visible to any thread that later loads `head` with `acquire` and walks into N.
  - **Failure**: another thread changed `head` between line 2 and line 4. `old_head` is auto-refreshed to the new real head; we loop, redo line 3 (relink N to the *new* head), and retry. N is still private, so relinking is free.

**Where ABA would bite:** suppose between line 2 and line 4, a concurrent *pop* on thread Q pops B (head becomes C), frees B, then a later push reuses B's exact address for a different node B'' and the stack ends up `head -> B'' -> ...` where `B''` lives at the *same address* as the freed B. Thread P's CAS at line 4 compares `head` against `old_head == B`'s address. If `head` now holds B'' at the same bits, **the CAS succeeds wrongly**, splicing N in front of B'' while believing it's in front of the original B — and B may be freed. Push is *less* prone to ABA than pop (push doesn't dereference `old_head`, it only compares it), but the failure is real whenever the head's address is recycled. The fix is the same: a tagged head so `(B, tag=7)` and `(B'', tag=9)` differ, plus hazard pointers so B isn't freed while P holds its address. This is why the earlier `delete old_head` in pop was flagged: *that very free* is what arms the ABA gun.

## Measured: lock-free vs locked under contention

Now the honest measurement, because lock-free is not magic and selling it as such has burned a lot of teams. The headline: lock-free wins *under contention*, and can *lose* without it. Here is a representative shape of results for a stack benchmark — N threads each doing tight push/pop pairs on one shared stack, on a typical multi-core x86 server. Treat these as order-of-magnitude and directional, not precise; the exact numbers depend wildly on the machine, allocator, and contention pattern, and you must measure your own.

| Threads | `std::mutex` stack (ops/s) | Treiber + hazard ptr (ops/s) | Winner |
| --- | --- | --- | --- |
| 1 | ~38 M | ~30 M | mutex (uncontended lock is cheap; CAS + HP fence isn't free) |
| 2 | ~12 M | ~22 M | lock-free |
| 4 | ~6 M | ~28 M | lock-free, widening |
| 8 | ~3 M | ~30 M | lock-free, big |
| 16 | ~1.8 M | ~26 M | lock-free, large; mutex convoys |

Read the trend, not the digits. At **one thread** the mutex often *wins*: an uncontended mutex is just a couple of atomic ops with no waiting, while the lock-free version pays for its CAS plus the hazard-pointer fence and retire bookkeeping on every operation. As threads climb, the mutex throughput *collapses* — threads convoy behind the lock, sleeping and waking, paying context-switch and cache-coherence costs — while the lock-free stack stays roughly flat-to-high because a winner always makes progress without anyone blocking. The crossover is usually somewhere between 2 and 4 threads.

![before and after comparison of throughput under contention showing a lock based stack plateauing as threads convoy versus a lock-free stack scaling two to four times higher](/imgs/blogs/compare-and-swap-and-building-lock-free-data-structures-7.png)

#### Worked example: how to measure this honestly

If you run this yourself, the mistakes that produce garbage numbers are predictable:

1. **Warm up.** The allocator, the branch predictor, and the JIT (on the JVM) all need a few hundred thousand iterations before steady state. Discard the first run.
2. **Run many times and report the distribution**, not one number. Concurrency benchmarks are noisy; report median and a spread, and say how many runs.
3. **Pin the contention level you care about.** "Lock-free is faster" is meaningless without "at 8 threads doing 100% stack ops." If your *real* workload touches the stack 1% of the time and does other work the rest, the contention is low and the mutex may well win. Benchmark your *actual* access pattern.
4. **Watch the allocator.** A naive `new`/`delete` per push/pop can make *allocation* the bottleneck, hiding the difference between mutex and lock-free entirely. Use a pool so you're measuring the structure, not `malloc`.
5. **Name the platform.** x86 has strong (TSO) memory ordering, so `acquire`/`release` are nearly free there; on ARM's weaker model the same code emits real barriers and the numbers shift. The memory model changes the answer — see [atomics and memory orderings](/blog/software-development/concurrency/atomics-and-memory-orderings-from-relaxed-to-seq-cst).

The takeaway from the numbers is not "lock-free is fast." It is: **lock-free trades a higher fixed per-operation cost for the absence of a contention cliff.** If you live on the flat part of the contention curve — low contention — you may be paying that fixed cost for nothing. If you live on the cliff — high contention on a hot structure — lock-free is the difference between scaling and not.

## Case studies / real-world

These structures are not academic curiosities; they are in the runtime you use every day.

- **Treiber's stack (R. Kent Treiber, IBM, 1986).** The original lock-free stack, from the IBM Almaden report *Systems Programming: Coping with Parallelism*. It is the canonical first lock-free structure precisely because both operations reduce to one head CAS, and it is where the ABA problem was first widely discussed in the context of pointer reuse. Nearly every lock-free tutorial — including this one — starts here.
- **The Michael-Scott queue (Maged Michael and Michael Scott, PODC 1996), *"Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms."*** This paper's two-CAS-with-helping queue became the industry standard FIFO. Its direct descendant is **`java.util.concurrent.ConcurrentLinkedQueue`** in the JDK, written by Doug Lea — a near-line-for-line implementation of the algorithm, relying on the JVM's GC to dodge the reclamation problem. When you use `ConcurrentLinkedQueue`, you are running 1996's algorithm. The same authors later introduced **hazard pointers** (Michael, 2004, *"Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"*) precisely to solve the reclamation problem their queue exposed.
- **The LMAX Disruptor (2011).** A high-performance ring buffer at the heart of the LMAX financial exchange, famous for processing millions of orders per second on a single thread's worth of contention by using a **fetch-add** sequence claim instead of a CAS loop, plus mechanical-sympathy cache padding to kill false sharing (covered in [cache coherence, MESI, and false sharing](/blog/software-development/concurrency/cache-coherence-mesi-and-false-sharing)). It is the production proof that the cheapest atomic (fetch-add, which never retries) beats CAS when the operation is a commutative slot claim. The published throughput figures were eye-opening at the time and drove a lot of interest in lock-free ring buffers.
- **The Linux kernel's RCU.** Read-copy-update is lock-free (lock-*less*, even) on the read side and is used pervasively in the kernel for read-mostly data like the dentry cache and routing tables. It is the most successful production reclamation scheme in existence and the reason the kernel can scale read-heavy structures to hundreds of cores. Paul McKenney's work on RCU is the definitive reference.

The throughput numbers cited for the Disruptor and the scaling claims for RCU are well-publicized but vary by workload and hardware generation; treat any single figure as approximate and representative rather than a guarantee.

## When to reach for this (and when not to)

Here is the decisive part, because the most important lock-free skill is *not building one*.

**Reach for lock-free when:**

- You have **measured** a specific structure as a contention bottleneck — the profile shows threads convoying on one lock, throughput plateauing or dropping as you add cores, and the structure is on the hot path. Measure first; "it might contend" is not a reason.
- The operation genuinely reduces to **one atomic publish** — a stack head, a queue tail/head, a per-bucket slot, a sequence counter. If your operation needs to atomically update several words, lock-free is going to be brutal; reconsider.
- You need **progress under stall** — a real-time or low-latency system where one thread being descheduled (or page-faulting, or hitting a GC pause on another thread) must not freeze the whole structure. This is the irreplaceable property of lock-free: a mutex cannot give it to you.
- You are on a **GC'd runtime** (Java, Go, C#) or can use a vetted reclamation library (`crossbeam-epoch`, `folly::ConcurrentHashMap`, C++26 `std::hazard_pointer`). This removes the hardest 80%.

**Do not reach for lock-free when:**

- The lock is **not your bottleneck.** This is the common case. A mutex held for a few nanoseconds, contended rarely, is faster *and* simpler than lock-free. Adding lock-free here makes the code slower (per-op overhead) and far harder to maintain, for no benefit. Profile before you reach.
- You would be **hand-rolling reclamation in C++/Rust** without a tested scheme. The probability of shipping a use-after-free or an ABA bug is high, the bugs are nondeterministic and nearly impossible to reproduce, and they corrupt memory rather than failing cleanly. Use a library.
- The operation **touches multiple words** or needs a cross-element invariant. Lock-free multi-word updates require techniques like multi-word CAS or software transactional memory ([software transactional memory and optimistic concurrency](/blog/software-development/concurrency/software-transactional-memory-and-optimistic-concurrency)), which are their own can of worms. A lock is usually the right call.
- A **library already has it.** `ConcurrentLinkedQueue`, `ConcurrentHashMap`, Go channels, `crossbeam` queues, `folly` — these are written by experts, tested for years, and handle reclamation. **The overwhelmingly correct default is to use the library, not build your own.** You build your own to *learn*, or in the rare case where you have a profiled need a library can't meet. Treat hand-rolled lock-free in production code as a red flag in review until proven necessary.

The blunt rule: **almost always, use the library.** The reason to understand the internals — the reason for this entire post — is so you can *choose* the right library structure, reason about its guarantees, debug it when it misbehaves, and recognize the rare case that justifies building one. Understanding lock-free is essential; *writing* it from scratch in production almost never is.

For the full decision framework across every concurrency model — when lock-free beats a mutex beats a channel beats an actor — see the capstone, [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).

A note on Python, since people ask: lock-free CAS-loop structures are largely *not* a Python concern at the application level, because the GIL serializes bytecode and `asyncio` is single-threaded — the contention model is entirely different. If you are reasoning about Python concurrency, the relevant mechanism is the GIL, not CAS; see [the GIL explained: what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs). Use Python lock-free thinking only when you drop into a C extension or a native library.

## Key takeaways

1. **Lock-free means the structure always makes progress, even if a thread stalls mid-operation** — at least one thread finishes in bounded steps regardless of scheduling. It is not the same as wait-free, which guarantees *every* thread finishes.
2. **Every lock-free mutation is a single atomic publish** — one CAS (or fetch-add) that transitions the structure from one valid state directly to another, with no observable half-updated state. Find the one word whose swap publishes your whole change; stage everything else privately first.
3. **The CAS loop is the universal idiom**: read current, compute new, compare-and-swap; on failure, someone else won the race, so re-read and retry. A failed CAS is a signal, never an error.
4. **The Treiber stack** publishes the head in one CAS; **the Michael-Scott queue** splits enqueue into link-then-advance-tail and uses *helping* so a stalled thread never freezes the queue.
5. **ABA bites when a value is a recycled identity** — a freed-then-reused pointer fools a value-comparing CAS. Fix the logical ABA with a tagged/versioned pointer plus double-width CAS; fix the use-after-free with hazard pointers or another reclamation scheme.
6. **Memory reclamation is the hard 80%, not the algorithm.** In a non-GC language, "when can I free this node?" is the real problem; hazard pointers, epochs, and RCU are the answers, each with a different bargain. A GC solves it for you, which is why lock-free is far easier on the JVM.
7. **Helping is what makes it lock-free**, not just an optimization: any thread completes another's half-done operation, so no thread's progress depends on a specific other thread waking up.
8. **Measure before and after.** Lock-free trades a higher fixed per-op cost for no contention cliff — it wins under heavy contention and can *lose* at low contention. Warm up, run many times, name your platform's memory model.
9. **Almost always use the library.** Hand-rolled lock-free in production is a red flag until a profile proves a library can't meet the need. Understand the internals so you can choose and debug the right one — not so you reinvent it.

## Further reading

- **Maurice Herlihy and Nir Shavit, *The Art of Multiprocessor Programming*** — the definitive textbook on lock-free and wait-free structures, progress conditions, and the universality of consensus. The Treiber stack, the Michael-Scott queue, and the ABA problem all live here with full proofs.
- **Maged Michael and Michael Scott, *"Simple, Fast, and Practical Non-Blocking and Blocking Concurrent Queue Algorithms"* (PODC 1996)** — the original Michael-Scott queue paper; short and readable, and the source of `ConcurrentLinkedQueue`.
- **Maged Michael, *"Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects"* (IEEE TPDS 2004)** — the canonical reclamation scheme, now standardized in C++26.
- **R. Kent Treiber, *"Systems Programming: Coping with Parallelism"* (IBM Research Report RJ 5118, 1986)** — the original lock-free stack.
- **Anthony Williams, *C++ Concurrency in Action* (2nd ed.)** — the practitioner's guide to `std::atomic`, memory orderings, and building lock-free structures correctly in modern C++.
- **Paul McKenney, *"Is Parallel Programming Hard, And, If So, What Can You Do About It?"*** — the deep dive on RCU and real-world reclamation, free online.
- **Preshing on Programming (Jeff Preshing's blog)** — exceptionally clear articles on lock-free programming, memory ordering, and the ABA problem, with runnable code.
- Within this series: [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), [how a lock is built: test-and-set, CAS, and spinlocks](/blog/software-development/concurrency/how-a-lock-is-built-test-and-set-cas-and-spinlocks), [the progress hierarchy: blocking, lock-free, and wait-free](/blog/software-development/concurrency/the-progress-hierarchy-blocking-lock-free-and-wait-free), [the ABA problem, TOCTOU, and torn reads](/blog/software-development/concurrency/the-aba-problem-toctou-and-torn-reads), [memory reclamation: hazard pointers, epochs, and RCU](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu), and the capstone [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
