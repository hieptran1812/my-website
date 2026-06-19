---
title: "The ABA Problem, TOCTOU, and Torn Reads"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "The subtle concurrency hazards that survive a naive fix: a CAS that succeeds yet corrupts, a security check that races, a 64-bit read that tears, and a singleton that publishes half-built."
tags:
  [
    "concurrency",
    "parallelism",
    "aba-problem",
    "toctou",
    "torn-read",
    "lock-free",
    "cas",
    "double-checked-locking",
    "memory-model",
    "systems-programming",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-aba-problem-toctou-and-torn-reads-1.png"
---

A few years ago I watched a lock-free stack pass every test in CI, run clean for a week in staging, and then drop nodes in production roughly once a day under load. The pop operation used a textbook compare-and-swap loop. The CAS returned `true` — it *succeeded* — and yet the stack ended up pointing at a node that had been freed, reallocated, and handed to a different part of the program. The bug was not a missing lock. It was not a data race in the strict sense; the access was atomic. The operation did exactly what it was told. The problem was that "the head pointer still equals the value I read a moment ago" turned out to be a lie, even though the bits matched exactly. The head had gone from node A to node B and back to a *recycled* node A while our thread was asleep. The CAS could not tell the difference, because a pointer is just an address, and an address recycled is bit-for-bit the address you remembered.

That is the ABA problem, and it is the headline member of a family of hazards I want to dissect in this post. They share a personality: each one survives the fix you reach for first. You add an atomic, you escape the obvious data race, and you feel safe — but the operation as a *whole* is still not atomic, or the value you compared is no longer the value you thought, or the thing you read is wider than the unit the hardware can move in one shot. These bugs are rare, timing-dependent, and brutal to reproduce. They are the ones that make a senior engineer distrust a passing test suite.

![A timeline showing thread one reading node A then thread two popping A pushing B popping B and recycling A so thread one's compare and swap on A succeeds wrongly and corrupts the stack](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-1.png)

By the end of this post you will be able to recognize four hazards on sight — the ABA problem in a CAS loop, time-of-check-to-time-of-use (TOCTOU) races in file and resource code, torn reads and writes of wide values, and lost updates, including the broken double-checked locking idiom that ties them together — and you will know the specific fix each one demands, in idiomatic code, in more than one language. This post sits in the [Concurrency & Parallelism series](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), and it builds directly on [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition). The recurring frame of the whole series applies here too: the hazard is shared mutable state plus nondeterministic scheduling, and the cure is to establish a happens-before order over every access and pick the cheapest mechanism that buys it. These four bugs are what happens when you *think* you bought that order but did not.

## The ABA problem: when "unchanged" is a lie

Start with the mechanism, because the whole point of ABA is that it looks impossible until you see exactly why CAS is blind to it.

Compare-and-swap is the workhorse of lock-free code. The operation `CAS(addr, expected, new)` does three things as one indivisible step the hardware guarantees: it reads `*addr`, compares it to `expected`, and — only if they are equal — writes `new` and reports success. On x86 this is the single `lock cmpxchg` instruction; on ARM and other weakly ordered machines it is a load-linked / store-conditional pair (`ldaxr` / `stlxr`) that retries if anything touched the line in between. Either way, the atomic guarantee is precise and narrow: *the read-compare-write happens with no other write to that location interleaved between the read and the write of this one CAS.*

Read that guarantee carefully, because the ABA bug lives in the gap it leaves. CAS guarantees nothing about what happened *before* you called it. If you read a value V at time t1, do some work, and then call `CAS(addr, V, new)` at time t3, the CAS only checks that `*addr == V` at t3. It cannot see the history between t1 and t3. If the location went V → W → V in that window, the CAS sees V at t3, concludes "nothing changed," and succeeds — even though everything changed and changed back.

For a counter that wraps, this is usually harmless: 5 → 6 → 5 and a CAS expecting 5 is fine, because the *meaning* of the value 5 did not change. ABA turns lethal when the value is a *pointer* and the memory it points to has been recycled, because then the bits are identical but the meaning is not.

#### Worked example: popping a recycled node

Here is the classic Treiber stack pop. The stack is a singly linked list; `head` points at the top node; popping means reading `head`, reading `head.next`, and swinging `head` to that next node with a CAS. Walk the interleaving instruction by instruction:

1. **T1** reads `head` and gets node **A**. It reads `A.next` and gets node **B**. It now intends to do `CAS(head, A, B)`. Then the scheduler preempts T1. It is holding the values A and B in registers and is about to commit.
2. **T2** runs to completion. It pops A (so `head` becomes B). It pops B (so `head` becomes whatever was under B, say `null`). It then *frees* A back to the allocator.
3. **T2** does a fresh `push`. The allocator, being a good allocator, hands back the *same address* it just freed — node A. T2 pushes this recycled A. Now `head` equals A again, but `A.next` now points somewhere completely different — at the old top, not B.
4. **T1** resumes. It executes `CAS(head, A, B)`. The current `head` *is* A (bit-for-bit), so the CAS **succeeds**. It sets `head = B`.

But B was popped and possibly freed in step 2. The stack's head now points at a node that is gone, or at a node that belongs to some other data structure, or at freed memory. We have corrupted the stack with a CAS that returned success. No data race tool flags this; every individual access was a well-formed atomic operation. The figure in the intro traces exactly this t1-through-t6 sequence.

The deep reason, stated once so it sticks: **CAS verifies a value, not a version.** Identity of the *value* is not identity of the *thing*. When a pointer is recycled, value-identity and thing-identity come apart, and CAS — which only ever knew about values — cannot tell.

There is one architecture where ABA *almost* hides, and the difference is worth a paragraph because it trips people up. On ARM, POWER, and RISC-V, the primitive is not a single `cmpxchg` but a **load-linked / store-conditional** (LL/SC) pair. `ldaxr` reserves the cache line and remembers it; `stlxr` commits the store only if *nothing has touched that line* since the reservation. Because LL/SC watches the *line*, not the *value*, a naive intuition says it should catch ABA — surely the A → B → A churn touched the line and would void the reservation. Sometimes it does. But you cannot rely on it: most LL/SC users wrap the pair in a retry loop that re-loads the value and re-issues the SC, and the reservation can be lost and re-acquired around the very window where ABA happens, so the high-level CAS loop built on LL/SC is exactly as ABA-prone as the x86 `cmpxchg` version. Worse, LL/SC reservations are routinely lost for *unrelated* reasons — a context switch, an interrupt, a nearby store to the same line (false sharing of the reservation granule) — so robust code must loop anyway and cannot lean on the line-watching behavior for correctness. The portable mental model is the safe one: assume your CAS sees only the value, on every architecture, and defend against ABA explicitly. x86's `cmpxchg` makes this unavoidable (it genuinely compares only the value); ARM's LL/SC merely makes it *tempting* to assume otherwise, which is a trap.

One more subtlety about the window's size. People assume "T1 sleeps" means an OS preemption lasting milliseconds, and conclude ABA needs a thread to be descheduled for a long time. Not so. The window is just the instructions between T1's `load` of `head` and T1's `CAS` — on a modern out-of-order core, a single cache miss on `old->next` can stall T1 for hundreds of cycles while another core, hitting in its own cache, races through a full pop-pop-recycle-push. The window can be *tens of nanoseconds*. That is why ABA shows up under high contention even when no thread is formally preempted: the hazard lives in microarchitectural stalls, not just scheduler decisions.

## Fixing ABA: tag the pointer, defer the free, or widen the swap

There are three production-grade families of fixes, and they correspond to three different ways of restoring the missing information. I will take them in order of how often you should reach for them.

![A before and after diagram contrasting a plain pointer compare and swap that is blind to a recycled A against a tagged pointer whose version counter makes the recycled A compare unequal so the stale swap fails and retries](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-2.png)

**Tagged pointers (the ABA counter).** Pack a monotonic counter next to the pointer and CAS the *pair* atomically. Every successful update bumps the counter. Now the value you compare is `(pointer, tag)`, and even if the pointer recycles back to A, the tag has advanced — so the stale CAS sees `(A, 7)` versus the live `(A, 9)`, finds them unequal, fails, and retries. The before/after figure above shows exactly this: the bits of the pointer match, but the version breaks the tie. This is the cheapest, most portable fix, and it is the right default when your pointer plus a counter fits in a double-word the hardware can CAS atomically.

On x86-64 you have two clean ways to make the pair atomic. The first is `lock cmpxchg16b`, a 128-bit compare-and-swap (double-width CAS, DWCAS), which swaps a full 16-byte struct of `{pointer, counter}`. The second, on platforms where pointers only use the low 48 bits, is *pointer tagging*: stash the counter in the unused high 16 bits of the pointer itself and use an ordinary 64-bit CAS. Tagging is faster and needs no special instruction, but the counter is small (16 bits wraps after 65,536 updates) and it relies on the high bits truly being free, which future CPUs with larger virtual address spaces can take away. Prefer DWCAS when you can afford it.

Here is the bug and the fix in C++ with `std::atomic`. First the broken, ABA-vulnerable pop:

```cpp
// BROKEN: plain pointer CAS, vulnerable to ABA on a recycled node.
#include <atomic>

struct Node { int value; Node* next; };

class BrokenStack {
  std::atomic<Node*> head_{nullptr};
public:
  void push(Node* n) {
    n->next = head_.load(std::memory_order_relaxed);
    while (!head_.compare_exchange_weak(
        n->next, n, std::memory_order_release, std::memory_order_relaxed)) {}
  }
  Node* pop() {
    Node* old = head_.load(std::memory_order_acquire);
    while (old && !head_.compare_exchange_weak(
        old, old->next, std::memory_order_acquire, std::memory_order_relaxed)) {}
    return old; // ABA: 'old' may have been freed and recycled between load and CAS
  }
};
```

The fixed version carries a tag in a double-word and CASes the pair. On a 64-bit target with the 16-byte atomic available, the `TaggedHead` struct is exactly the width `cmpxchg16b` swaps:

```cpp
// FIXED: tagged pointer (ABA counter) swapped as a double-word.
#include <atomic>
#include <cstdint>

struct Node { int value; Node* next; };

struct TaggedHead {
  Node* ptr;
  uintptr_t tag;   // bumped on every successful update; breaks ABA
};

class TaggedStack {
  // 16-byte alignment lets the compiler emit lock cmpxchg16b for the pair.
  alignas(16) std::atomic<TaggedHead> head_{ TaggedHead{nullptr, 0} };
public:
  void push(Node* n) {
    TaggedHead cur = head_.load(std::memory_order_relaxed);
    TaggedHead next;
    do {
      n->next = cur.ptr;
      next = TaggedHead{ n, cur.tag + 1 };
    } while (!head_.compare_exchange_weak(
        cur, next, std::memory_order_release, std::memory_order_relaxed));
  }
  Node* pop() {
    TaggedHead cur = head_.load(std::memory_order_acquire);
    TaggedHead next;
    do {
      if (!cur.ptr) return nullptr;
      next = TaggedHead{ cur.ptr->next, cur.tag + 1 };
    } while (!head_.compare_exchange_weak(
        cur, next, std::memory_order_acquire, std::memory_order_relaxed));
    return cur.ptr; // tag advanced on every update, so a recycled ptr no longer matches
  }
};
```

A note many people miss: `std::atomic<TaggedHead>` is only lock-free if your platform actually has the double-width CAS. Check `head_.is_lock_free()` at runtime — if it returns `false`, your compiler quietly wrapped the 16-byte struct in a hidden mutex, and you have a locked stack pretending to be lock-free. On GCC and Clang you also need to compile with `-mcx16` to get `cmpxchg16b` emitted instead of the locked fallback.

In Rust the same fix uses `AtomicU128` semantics through the `crossbeam` ecosystem in practice, but you can show the principle directly with a packed `u128` and `compare_exchange` where the platform supports it:

```rust
// FIXED (Rust): pointer + tag packed into a u128 and swapped atomically.
use std::sync::atomic::{AtomicU128, Ordering};

#[inline]
fn pack(ptr: *mut Node, tag: u64) -> u128 {
    ((ptr as u128) << 64) | (tag as u128)
}
#[inline]
fn unpack(word: u128) -> (*mut Node, u64) {
    (((word >> 64) as u64) as *mut Node, word as u64)
}

struct Node { value: i32, next: *mut Node }

struct TaggedStack { head: AtomicU128 }

impl TaggedStack {
    fn pop(&self) -> *mut Node {
        let mut cur = self.head.load(Ordering::Acquire);
        loop {
            let (ptr, tag) = unpack(cur);
            if ptr.is_null() { return std::ptr::null_mut(); }
            let next = unsafe { (*ptr).next };
            let desired = pack(next, tag.wrapping_add(1)); // tag bump defeats ABA
            match self.head.compare_exchange_weak(
                cur, desired, Ordering::AcqRel, Ordering::Acquire) {
                Ok(_) => return ptr,
                Err(observed) => cur = observed, // retry with the value we actually saw
            }
        }
    }
}
```

(`AtomicU128` is stable on recent Rust toolchains for targets with 128-bit atomics; in older code people reached for `crossbeam_utils::atomic::AtomicCell` or `AtomicTagged` crates. The shape is the same: pack, bump, CAS the pair.)

**Hazard pointers, epochs, and RCU (defer the free).** The deeper cure attacks the *cause* — node recycling — rather than the *symptom*. If a node A can never be freed and recycled while any other thread might still be holding a reference to it, then A cannot reappear under T1's feet, and ABA simply cannot happen. That is exactly what safe memory reclamation schemes guarantee. With **hazard pointers**, a thread publishes the pointer it is about to dereference into a per-thread "I am using this" slot; a thread that wants to free a node first scans every hazard slot and defers the free if anyone has it announced. With **epoch-based reclamation** or **RCU** (read-copy-update, the Linux kernel's workhorse), readers run inside a lightweight critical section and reclamation waits for a grace period in which every reader has moved on. These remove ABA *and* the use-after-free that lurks beneath it, at the cost of more machinery. They are the subject of a dedicated post — [memory reclamation: hazard pointers, epochs, and RCU](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu) — so I will not re-derive them here, but the one-line takeaway is: *tagged pointers stop the stale CAS from succeeding; reclamation schemes stop the node from being recyclable in the first place.* The second is strictly more powerful, because a tag does not save you from dereferencing freed memory while you read `old->next`.

The distinction between those two failure modes matters enough to make it concrete. The tagged-pointer fix prevents the *wrong CAS from succeeding*, but in the broken pop the line `next = TaggedHead{ cur.ptr->next, cur.tag + 1 }` *dereferences* `cur.ptr` to read `next` — and if another thread freed that node a nanosecond earlier, you have already read through a dangling pointer before the CAS ever runs. A tag does nothing about that read; it only catches the subsequent CAS. So a purely tag-based lock-free stack is safe *only* if nodes are never actually returned to the allocator while the structure is live (for example, you keep a private free-list of nodes that the OS never reclaims, so the address is always valid memory even when stale). The moment you call real `free`/`delete` on a popped node, a tag is not enough — you need a reclamation scheme to guarantee no thread is mid-dereference. This is the single most common mistake in hand-rolled lock-free code: a tagged pointer that defeats ABA but still reads through freed memory under load.

Here is the skeleton of the hazard-pointer cure, in C++, to make "defer the free" tangible. The reader *announces* the node it is about to touch; the reclaimer scans announcements before freeing:

```cpp
// Hazard-pointer pop (sketch): announce, validate, then it is safe to deref.
#include <atomic>

extern std::atomic<Node*> g_hazard[kMaxThreads]; // one announce slot per thread

Node* pop_hp(std::atomic<Node*>& head, int tid) {
  Node* old;
  do {
    old = head.load(std::memory_order_acquire);
    if (!old) return nullptr;
    g_hazard[tid].store(old, std::memory_order_seq_cst); // announce intent
    // re-validate: if head moved, our announcement may be stale, retry
  } while (old != head.load(std::memory_order_acquire) ||
           !head.compare_exchange_weak(old, old->next,
               std::memory_order_acq_rel, std::memory_order_acquire));
  g_hazard[tid].store(nullptr, std::memory_order_release); // done with it
  return old; // a reclaimer scanning g_hazard[] will defer freeing this node
}
```

The reclaimer, before it actually frees a retired node, walks `g_hazard[]` across all threads; if any slot still points at the node, it puts it on a deferred list and frees it later. No node any reader has announced can be reclaimed, so the address cannot be recycled under a reader's feet, so ABA — and the use-after-free beneath it — cannot occur. The cost is the per-node announce/scan; the benefit is correctness even when you call real `free`. The full treatment, including epochs and RCU's grace periods, lives in the dedicated reclamation post.

**Double-width CAS (DWCAS).** Already met above: it is the *instruction* that makes the tagged-pointer fix possible when pointer plus counter exceeds a single word. It is not a separate strategy so much as the hardware that the tagged-pointer strategy stands on. Worth naming because interviewers ask, and because `is_lock_free()` returning false is a silent trap.

The taxonomy figure later in this post collects all of these under two roots — make it atomic, or version it — so you can keep them straight.

## TOCTOU: the gap between checking and using

ABA is a race inside one data structure. TOCTOU — time-of-check-to-time-of-use — is the same *shape* of bug at the level of the operating system and the file system. You check a condition, the condition changes, and then you act on the now-stale check. The check and the act are two separate operations with a window between them, and an adversary (or just another thread, or another process) gets to run in that window.

![A timeline showing a process calling access on a path that passes the check then an attacker swapping the path to a symlink before the process calls open so it opens the wrong file and writes as root](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-3.png)

The canonical example is the `access()` + `open()` security hole, and it has been a classic source of local privilege escalation for decades. A setuid program — one that runs with elevated privileges but acts on behalf of a normal user — wants to be polite: before it opens a file as root, it checks whether the *real* (unprivileged) user is allowed to access that path, using `access()`, which checks against the real user ID rather than the effective one. If the check passes, it opens and writes. The intent is sound. The implementation is a textbook TOCTOU race:

```c
/* BROKEN: classic access()/open() TOCTOU in a setuid program. */
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>

int write_user_file(const char *path, const char *data, size_t len) {
    /* TIME OF CHECK: does the *real* user own/have access to this path? */
    if (access(path, W_OK) != 0) {
        return -1;                 /* real user not allowed: refuse */
    }
    /* ---- attacker's window opens here ---- */
    /* TIME OF USE: open as the *effective* user (root) */
    int fd = open(path, O_WRONLY | O_TRUNC);   /* may now point elsewhere */
    if (fd < 0) return -1;
    write(fd, data, len);
    close(fd);
    return 0;
}
```

Between `access()` and `open()`, the attacker — who controls the directory `path` lives in — deletes the innocuous file they pointed `path` at and replaces it with a *symlink* to `/etc/passwd` (or `/etc/shadow`, or root's `authorized_keys`). `access()` checked the harmless file and said "yes, the real user may write here." `open()`, running as root, follows the symlink and opens `/etc/passwd`. The program writes attacker-controlled data into a root-owned file. That is a local root exploit, and the timeline figure above traces the swap landing precisely in the t2-to-t4 gap.

The mechanism is identical to ABA in its essence: a *name* (here, the path string; there, the pointer value) was resolved to a *thing* (a file; a node) at check time, but the binding from name to thing changed before use time, and the code trusted the stale binding. Path-based file APIs are pervasively TOCTOU-prone because a path is re-resolved on *every* call — each `open`, `stat`, `chmod` walks the directory tree fresh, so an attacker who can swap a component mid-sequence gets a new binding each time.

#### Worked example: the check that lies

Concretely, suppose `path` is `/tmp/userdir/report.txt` and the attacker owns `/tmp/userdir`. The sequence is:

1. **t1** — program calls `access("/tmp/userdir/report.txt", W_OK)`. The real file is a normal file the user owns. Check returns 0 (allowed).
2. **t2** — attacker runs `rm /tmp/userdir/report.txt; ln -s /etc/passwd /tmp/userdir/report.txt`. The name now resolves to a different thing.
3. **t3** — program calls `open("/tmp/userdir/report.txt", O_WRONLY|O_TRUNC)` as root, follows the symlink, and truncates and opens `/etc/passwd`.
4. **t4** — program writes its payload into `/etc/passwd`.

The attacker does not need to win this race on the first try. They run a tight loop swapping the file back and forth and run the victim program thousands of times; even a one-in-ten-thousand hit rate is a guaranteed exploit given a script. **TOCTOU windows do not need to be likely — they need only be possible**, because the attacker controls the retry count.

## Fixing TOCTOU: bind once, then operate on the binding

The cure is to collapse the check and the use into one atomic step, or to stop re-resolving the name. Three idioms, strongest first.

**Operate on a handle, not a name.** Resolve the path to a file descriptor *once*, then do everything through that fd. A descriptor is a stable kernel handle: once `open` returns it, it points at a specific inode and stays pointed there even if the path is later rewritten. So `open` first, then `fstat` the fd, then check the fd's attributes, then act on the fd. There is no second name resolution to attack:

```c
/* FIXED: open once to a handle, then verify and act on the descriptor. */
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

int write_user_file_safe(const char *path, const char *data, size_t len,
                         uid_t real_uid) {
    /* Open without following a final symlink; do not create. */
    int fd = open(path, O_WRONLY | O_NOFOLLOW);
    if (fd < 0) return -1;

    struct stat st;
    if (fstat(fd, &st) != 0) { close(fd); return -1; }   /* checks THIS fd */

    /* Check ownership on the very inode we hold, not on a name we re-resolve. */
    if (st.st_uid != real_uid || !S_ISREG(st.st_mode)) {
        close(fd);                                        /* refuse politely */
        return -1;
    }
    write(fd, data, len);
    close(fd);
    return 0;
}
```

`O_NOFOLLOW` makes `open` refuse if the final path component is a symlink, killing the symlink swap outright; `fstat` checks the inode behind the descriptor we *already hold*, so there is no name to re-bind. For directory traversal you escalate to the `*at` family — `openat`, `fstatat`, `unlinkat` with `AT_SYMLINK_NOFOLLOW` — which resolve relative to a directory fd you pin, so no component of the path can be swapped under you. On Linux, `O_PATH` plus `openat2` with `RESOLVE_NO_SYMLINKS` / `RESOLVE_BENEATH` gives even stronger guarantees.

**Atomic check-and-act primitives.** Where the kernel offers a single call that both checks and acts, use it. `open(path, O_CREAT | O_EXCL)` atomically creates the file *only if it does not already exist* — there is no window between "check it's absent" and "create it," because the kernel does both under one inode lock. This is the correct way to create a lock file or a unique temp file; the naive `if (!exists(path)) create(path)` is a TOCTOU race. `mkstemp` exists for exactly this reason.

**Operate-then-handle-failure (capabilities / optimistic).** When you cannot collapse check and act, flip the order: *attempt the operation and handle the failure*, rather than check-then-do. Instead of "check the directory is empty, then delete it," call `rmdir` and handle `ENOTEMPTY`. Instead of "check the user has a quota, then write," write and handle the quota error. The operation itself is the atomic check. This is the same principle that makes a CAS loop correct: do not look-then-leap; leap atomically and retry on failure. Capability-based systems generalize this — you hold an unforgeable token that *is* the permission to act on a specific object, so there is no name to re-resolve and no separate check to race against.

A subtlety worth stating: TOCTOU is not only a security bug. Any "look at shared state, then act on what you saw" pattern across threads is a TOCTOU race — `if (map.containsKey(k)) map.get(k)` can return null if another thread removed `k` between the two calls. The fix is the same: use the atomic compound operation (`map.computeIfAbsent(k, ...)`, `getOrDefault`, `putIfAbsent`) instead of check-then-act. We covered the threaded version of this under [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition); TOCTOU is its OS-and-filesystem-flavored cousin.

The threaded form is worth a code sketch because it is so easy to write and so easy to miss in review. Even on a *thread-safe* `ConcurrentHashMap`, the compound "check then act" is not atomic — each individual call is safe, but the gap between them is wide open:

```java
// BROKEN: each call is thread-safe, but the check-then-act compound is not.
ConcurrentHashMap<String, Connection> pool = new ConcurrentHashMap<>();

Connection getOrOpen(String host) {
    if (!pool.containsKey(host)) {          // TIME OF CHECK
        // another thread can insert here
        pool.put(host, openExpensive(host)); // TIME OF USE: may open twice, leak one
    }
    return pool.get(host);                    // may even return the other thread's value
}

// FIXED: one atomic compound primitive closes the window.
Connection getOrOpenSafe(String host) {
    return pool.computeIfAbsent(host, this::openExpensive); // check-and-act, atomic
}
```

The broken version can open the expensive connection twice under contention, leaking one and possibly handing different callers different connections to the same host. `computeIfAbsent` performs the test-and-insert as one atomic operation the map serializes internally — exactly the "collapse check and act into one step" cure, just spelled in a library method instead of a kernel call. The lesson generalizes far past maps: *thread-safe building blocks do not make compound operations over them thread-safe.* A sequence of individually atomic calls is not itself atomic, and that gap is a TOCTOU race every time.

## The hazard family at a glance

Before we get to torn reads, it helps to see all four hazards side by side, because the fixes rhyme. Every one of them is a failure to keep an operation atomic, and every fix restores atomicity by a different route — versioning, a single compound primitive, the right access width, or a lock.

![A matrix mapping four hazards which are ABA in compare and swap, time of check to time of use, torn read or write, and lost update, to what goes wrong and the specific fix for each](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-4.png)

| Hazard | The lie it tells | Why a naive fix fails | The real fix |
| --- | --- | --- | --- |
| ABA in CAS | "the value is unchanged" | the value is the same but the thing was recycled | tagged pointer, hazard pointer, DWCAS |
| TOCTOU | "the check still holds" | the binding changed between check and use | operate on a handle, atomic check-and-act |
| Torn read/write | "I read one value" | the value is wider than the atomic unit | atomic access, alignment, word-sized values |
| Lost update | "my write landed" | another write overwrote it after my read | CAS loop, or hold a lock across read-modify-write |

The matrix figure above carries the same four rows. Keep it in mind as a checklist: any time you have shared state and a multi-step operation, ask which of these four windows you have left open. Now, torn reads — the one that is purely about hardware width.

## Torn reads and writes: when a value is wider than the bus

A torn read (or torn write) happens when you read or write a value that is *wider than the largest unit the hardware moves atomically*, so the access is split into pieces and another thread observes the value mid-update — half old, half new. This is not a logic bug in your code at all; it is the gap between the size of your data and the size of an atomic access on your machine.

![A before and after diagram showing a 64-bit value split into a high word with the new value and a low word with the old value producing a garbage read versus an aligned atomic load that the reader sees as either fully old or fully new](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-5.png)

The cleanest mechanism to picture is a 64-bit value on a 32-bit machine. The CPU can move 32 bits in one bus transaction but needs *two* transactions for 64 bits: one for the low word, one for the high word. Suppose a counter holds `0x0000_0000_FFFF_FFFF` and a writer increments it to `0x0000_0001_0000_0000`. That increment writes the low word (`FFFF_FFFF` → `0000_0000`) and the high word (`0000_0000` → `0000_0001`) as two separate stores. A reader that loads the two words *between* those two stores can see the new high word and the old low word — `0x0000_0001_FFFF_FFFF` — a value that *never existed* in the program's logic. The before/after figure above shows exactly this split: high word new, low word old, garbage in the middle. The atomic load on the right is indivisible: the reader sees either the full old value or the full new value, never a blend.

The same tearing happens for any value wider than the atomic unit, and structs are the common real-world case even on 64-bit machines. A `struct { double x, y; }` is 16 bytes; reading or writing it is two 8-byte accesses, and a concurrent reader can see the new `x` paired with the old `y`. A `std::string` or a fat pointer (pointer + length) is multiple words; assigning it under concurrency tears into a pointer that does not match its length — an immediate crash or buffer overrun. This is why "just read the field, it's only one variable" is wrong reasoning: *atomicity is a property of the access width and alignment, not of how many variable names you used.*

#### Worked example: the torn struct timestamp

Suppose a monitoring thread publishes a `struct timespec { long sec; long nsec; }` — two 64-bit fields, 16 bytes — by plain assignment, and another thread reads it to compute elapsed time:

1. Publisher holds `{sec=100, nsec=999_999_999}` (just before a second rollover).
2. Publisher assigns `{sec=101, nsec=0}`. The compiler emits a store of `sec` then a store of `nsec` (or the reverse — order is not guaranteed).
3. Reader loads `sec` = 101 (new), then loads `nsec` = 999_999_999 (old, not yet overwritten).
4. Reader computes a timestamp of `101.999999999` — almost a full second *into the future* relative to any value that actually existed.

If that timestamp drives a timeout, you just fired a timer a second early; if it drives a billing meter, you over-charged. The value `101.999999999` was never assigned by anyone. It is an artifact of reading a 16-byte struct as two 8-byte loads. The fix is to make the publish atomic — a lock around both fields, or an atomic of a word-sized representation, or pack the two fields into one 64-bit value if their ranges allow.

There is a crucial alignment wrinkle even on machines where the value *fits* in one access. On x86, an aligned 64-bit load or store is atomic — the hardware guarantees it. But a *mis-aligned* 64-bit access that straddles a cache-line boundary (a "split lock") is **not** guaranteed atomic and can tear, in addition to being 10–100× slower because the CPU must lock the bus across two lines. So even a single `int64_t` can tear if it is packed at a misaligned offset inside a struct. Natural alignment is not just a performance nicety here; it is a *correctness* requirement for tear-free access.

The torn *write* deserves equal billing, because it is sneakier than the torn read. When a writer stores a wide value non-atomically, the value spends a brief interval *physically inconsistent in memory* — and any reader, atomic or not, that loads during that interval sees the inconsistency. The reader does not have to be doing anything wrong. Consider a fat pointer published as `(ptr, len)` — a `std::string_view`, a Go slice header, a Rust `&[T]`. The writer assigns a new view: store the new `ptr`, then store the new `len` (or vice versa). A reader that loads `ptr` (new, pointing at a 4-byte buffer) and `len` (old, say 1000) now holds a view that says "1000 bytes starting at a 4-byte buffer." Dereferencing it reads 996 bytes past the end of the allocation — a heap over-read, an information leak, or a segfault, from code that merely *read* a variable. This is why languages with fat pointers either forbid sharing them across threads without synchronization (Rust's `Send`/`Sync` bounds enforce it at compile time) or require an atomic/lock for the publish. The torn write turns a read into a memory-safety violation.

#### Worked example: the fat pointer over-read

Make the slice-header tear concrete with numbers:

1. Writer holds a view `(ptr = 0x7000, len = 1000)` over a 1000-byte buffer.
2. Writer rebinds the view to a fresh 4-byte buffer at `0x9000`: it stores `ptr = 0x9000`, and is *about* to store `len = 4`.
3. Reader loads the header *between* those two stores: it gets `ptr = 0x9000` (new) and `len = 1000` (old).
4. Reader iterates `for i in 0..len` reading `*(ptr + i)` — reading bytes `0x9000` through `0x93E7`, i.e. 996 bytes past the 4-byte allocation.

The view `(0x9000, 1000)` was never a state the writer intended; it is the torn cross-product of the new pointer and the old length. The fix is to publish the whole header atomically (a 16-byte atomic, or a single-word atomic pointer to an immutable header) or under a lock — the same word-size-or-lock dichotomy as before, but here the stakes are memory safety, not just a wrong number.

## Fixing torn access: atomics, alignment, word-sized values

Three remedies, and they compose:

**Use an atomic type.** The portable, correct answer is to declare the value as an atomic and let the language guarantee indivisible access. `std::atomic<uint64_t>` in C++, `AtomicLong` in Java, `atomic.Int64` in Go, `AtomicU64` in Rust — each guarantees the load and store are all-or-nothing, emitting a single locked instruction or, on 32-bit targets, a lock-cmpxchg8b or a hidden lock when the hardware lacks a native 64-bit atomic. Critically, in C and C++, a *plain* `int64_t` accessed from multiple threads without synchronization is a **data race**, which is undefined behavior — the compiler is free to tear it, reorder it, or assume it never changes. `std::atomic` is what tells the compiler "other threads touch this; emit a real, indivisible access." This is also why the Java memory model historically did *not* guarantee atomicity for plain non-`volatile` `long` and `double` — the spec explicitly allowed a 64-bit non-volatile write to be split into two 32-bit writes, so a concurrent reader could see a torn `double`. Marking the field `volatile` (or using `AtomicLong` / `AtomicLongFieldUpdater`) restores atomicity. We will see `volatile` again in the double-checked locking section; here it is buying you tear-free 64-bit access.

```java
// Java: a non-volatile long/double could legally tear under the JMM.
// 'volatile' (or AtomicLong) makes the 64-bit access atomic.
class Meter {
    volatile long balanceCents;          // tear-free reads and writes
    final java.util.concurrent.atomic.AtomicLong counter = new AtomicLong();

    void add(long delta) {
        counter.addAndGet(delta);        // atomic read-modify-write, no torn write
    }
}
```

```cpp
// C++: a plain int64_t shared across threads is a data race (UB) and may tear.
// std::atomic<uint64_t> guarantees indivisible, race-free access.
#include <atomic>
#include <cstdint>

std::atomic<uint64_t> g_counter{0};

void publish(uint64_t v) {
    g_counter.store(v, std::memory_order_relaxed);   // single, untorn store
}
uint64_t observe() {
    return g_counter.load(std::memory_order_relaxed); // single, untorn load
}
```

**Keep the shared value word-sized.** If the value fits in the machine word and is naturally aligned, an atomic access of that word is cheap and tear-free. When the data is naturally bigger than a word — a struct, a string — you have two choices: shrink it to fit (pack two `int32` ranges into one `int64`; replace the fat value with an index or a pointer to immutable data) or protect it with a lock. *Atomic pointer swap of immutable data* is the elegant pattern: never mutate the struct in place; build a new immutable copy, then atomically swap a single pointer to it. The pointer is word-sized so the swap cannot tear, and readers either see the whole old struct or the whole new one — which is precisely how RCU and copy-on-write publish wide state safely.

**Align everything, and watch for false sharing's evil twin.** Ensure 8-byte values sit on 8-byte boundaries (most compilers do this for you, but packed structs, serialized buffers, and FFI boundaries can defeat it). On x86, enable split-lock detection (`split_lock_detect`) in production to *catch* misaligned atomics that would otherwise tear silently and tank performance.

| Approach | Tear-free? | Cost | Use when |
| --- | --- | --- | --- |
| plain wide value, no sync | no | free, but a data race | never, for shared mutable data |
| atomic of word-sized T | yes | one locked instruction | the value fits in a word |
| lock around the value | yes | mutex acquire/release | the value is a wide struct |
| atomic swap of immutable copy | yes | a copy + a pointer CAS | reads dominate, writes rare |

## Double-checked locking: the bug that needs a barrier

Now the idiom that ties torn access, ordering, and lost updates into one infamous knot: **double-checked locking** (DCL). The goal is benign — lazily initialize a shared singleton, paying the lock cost only on the *first* access and skipping it on every subsequent read. The naive implementation is a famous trap, and understanding *why* it is broken is the best memory-model lesson in this whole post.

![A before and after diagram showing broken lazy initialization where the pointer is published before the object is constructed so a reader sees a half built object versus a fixed version where fields are written first then a release store publishes the pointer and an acquire load sees the full object](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-6.png)

Here is the broken version, in the shape that shipped in countless C++ and pre-Java-5 codebases:

```java
// BROKEN double-checked locking (pre-JSR-133 semantics, no volatile).
class Singleton {
    private static Singleton instance;          // NOT volatile: the bug

    static Singleton get() {
        if (instance == null) {                 // first check, no lock
            synchronized (Singleton.class) {
                if (instance == null) {         // second check, under lock
                    instance = new Singleton(); // <-- the dangerous store
                }
            }
        }
        return instance;
    }
}
```

The reasoning that makes it *look* right: take the lock only when `instance` is null; re-check under the lock so only one thread constructs; everyone else sails through the cheap first check. Where it breaks is the line `instance = new Singleton()`. That single line of source is *not* a single operation. It is three steps: (1) allocate memory for the object, (2) run the constructor to initialize its fields, (3) store the object's address into `instance`. Crucially, **the compiler and the hardware are allowed to reorder steps 2 and 3** — to publish the pointer *before* the fields are initialized — because within a single thread the reorder is invisible (you never observe a half-built object from the thread that built it). There is no happens-before edge forcing the field writes before the pointer write as seen by *another* thread.

So picture the failure: thread T1 enters `get`, takes the lock, and runs `instance = new Singleton()` — but the optimizer publishes the pointer (step 3) before running the constructor (step 2). Thread T2 calls `get`, hits the first check, sees `instance != null` (T1 already published the pointer), skips the lock entirely, and returns a pointer to an object **whose constructor has not finished running**. T2 reads uninitialized fields — zeros, garbage, a null inner reference — and crashes or corrupts data. The before/after figure above shows exactly this: on the broken side the pointer is published before the fields; a reader sees non-null early and uses a half-built object. No lock was missing. The mutual exclusion worked perfectly. What was missing was an *ordering guarantee* between the constructor's writes and the pointer's write, across threads.

This is the same family as a torn read — the reader observes a partially-updated value — but the mechanism is *memory reordering*, not access width. And it is the same family as TOCTOU — the first check is stale by the time it is used — but the staleness comes from the absence of a barrier. DCL done wrong is all three hazards wearing one disguise.

It is worth being precise about *who* does the reordering, because there are two independent culprits and both must be fenced. The **compiler** reorders at compile time: an optimizer that sees `tmp = alloc(); tmp->x = 1; instance = tmp;` is free to rewrite it as `instance = tmp; tmp->x = 1;` if it can prove the single-threaded result is unchanged — and it cannot see other threads, so it does. The **hardware** reorders at run time: even if the compiler emits the stores in program order, a weakly-ordered CPU (ARM, POWER) has a store buffer that can let the pointer store become visible to other cores *before* the field stores drain, because the architecture only promises that *each core sees its own* writes in order, not that other cores do. On x86's stronger TSO model, stores are not reordered with other stores, so the hardware half of this bug does not fire — which is exactly why broken DCL "worked" on x86 in testing and then corrupted data the first time it ran on an ARM server. The acquire-release pair fences *both* culprits: it forbids the compiler from moving the field writes past the release store, and it emits the hardware barrier (or uses a release-store instruction like ARM's `stlr`) that flushes the store buffer in the right order. This is the canonical reason "passed on my x86 laptop" is not evidence of correctness for ordering bugs.

The fix is to insert the missing happens-before edge with an **acquire-release** pair, which in Java is spelled `volatile`:

```java
// FIXED double-checked locking: volatile inserts the happens-before edge.
class Singleton {
    private static volatile Singleton instance;   // volatile is mandatory

    static Singleton get() {
        Singleton local = instance;               // one volatile read
        if (local == null) {
            synchronized (Singleton.class) {
                local = instance;
                if (local == null) {
                    local = new Singleton();
                    instance = local;             // volatile WRITE = release
                }
            }
        }
        return local;
    }
}
```

Why `volatile` fixes it, precisely: in the Java Memory Model (since JSR-133 / Java 5), a write to a `volatile` field is a **release** and a read of a `volatile` field is an **acquire**, and a `volatile` read that observes a `volatile` write establishes happens-before between them. That ordering edge forbids the constructor's field writes (which happen-before the `volatile` store of `instance` in program order) from being reordered *after* the store as seen by other threads, and it forbids the reader's subsequent field reads from floating *before* the `volatile` read of `instance`. So when T2's acquire-read sees a non-null pointer, the release semantics guarantee every field write the constructing thread did before publishing is visible. The `local` variable is a micro-optimization: it reads the volatile field once instead of twice. We go deep on the acquire/release mechanism in [memory barriers: acquire, release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences); here the lesson is narrower: *DCL is correct only with a release on the publish and an acquire on the read.*

In C++ the equivalent broken/fixed pair uses `std::atomic` with explicit memory orders — and the idiomatic modern answer is to not hand-roll DCL at all but use `std::call_once`:

```cpp
// FIXED (C++): explicit acquire/release on an atomic pointer.
#include <atomic>
#include <mutex>

class Singleton { /* ... */ };

std::atomic<Singleton*> g_instance{nullptr};
std::mutex g_mu;

Singleton* get() {
    Singleton* p = g_instance.load(std::memory_order_acquire);  // acquire
    if (p == nullptr) {
        std::lock_guard<std::mutex> lk(g_mu);
        p = g_instance.load(std::memory_order_relaxed);
        if (p == nullptr) {
            p = new Singleton();                                // construct
            g_instance.store(p, std::memory_order_release);     // release publish
        }
    }
    return p;
}
```

```cpp
// IDIOMATIC (C++11+): let the standard library handle ordering correctly.
#include <mutex>
class Singleton { /* ... */ };

Singleton& get() {
    static std::once_flag flag;
    static Singleton* instance = nullptr;
    std::call_once(flag, []{ instance = new Singleton(); });  // happens-before, once
    return *instance;
}
```

`std::call_once` (and the even simpler **function-local `static`**, which since C++11 is guaranteed thread-safe and lazily initialized exactly once with the right ordering) makes the whole DCL conversation moot for most singletons. The lesson generalizes: when a language gives you a correct primitive for lazy init, use it; hand-rolled DCL is the kind of cleverness that quietly breaks on a weakly-ordered CPU even after passing every test on x86. In Go you write `sync.Once`; in Rust, `std::sync::OnceLock` or `once_cell::sync::Lazy`. Each bakes the acquire-release in so you cannot get the barrier wrong.

| Language | Wrong way | Right way | What it inserts |
| --- | --- | --- | --- |
| Java | plain `static` field + DCL | `volatile` field + DCL, or `static` holder class | volatile = acquire/release |
| C++ | plain pointer + DCL | `std::atomic` acquire/release, or `std::call_once` | explicit memory orders |
| Go | `if x == nil { x = new() }` racing | `sync.Once.Do(...)` | once-and-happens-before |
| Rust | `static mut` + manual check (UB) | `OnceLock` / `Lazy` | safe init, ordered publish |

## Lost updates: the read-modify-write that drops a write

The last hazard is the simplest and the one underneath the others: the **lost update**. Two threads each read a value, each compute a new value from it, and each write back — and one of the writes silently vanishes because it was based on a stale read. This is the canonical `count++` race written large, and it is worth a paragraph because it is the *baseline* the other three hazards are variations on.

The mechanism is the read-modify-write interleaving. `balance += 100` is three operations: load `balance`, add 100, store `balance`. If T1 and T2 both run it starting from `balance == 500`:

1. T1 loads 500.
2. T2 loads 500 (before T1 stores).
3. T1 computes 600, stores 600.
4. T2 computes 600 (from its stale 500), stores 600.

The balance should be 700; it is 600. One deposit of 100 was *lost* — overwritten by a write computed from a value read before the first write landed. This is exactly the lost-update interleaving from [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition), and it is the reason atomics and locks exist. The two real fixes:

- **Hold a lock across the whole read-modify-write**, so steps 1–3 are one critical section no other thread can interleave into. Correct, simple, and the right default when contention is low.
- **Use an atomic read-modify-write primitive** — `fetch_add`, `addAndGet`, or a CAS loop — so the read and the write are one indivisible step the hardware serializes. Lock-free, and the only option if you are avoiding locks.

```go
// Go: the lost update, and two fixes.
import "sync"
import "sync/atomic"

// BROKEN: plain read-modify-write races, loses updates.
var balance int64
func depositBroken(n int64) { balance += n }  // load, add, store — not atomic

// FIX A: a lock makes the RMW one critical section.
var mu sync.Mutex
func depositLocked(n int64) { mu.Lock(); balance += n; mu.Unlock() }

// FIX B: an atomic RMW primitive serializes load-add-store in hardware.
var balanceAtomic int64
func depositAtomic(n int64) { atomic.AddInt64(&balanceAtomic, n) }
```

Note the relationship to ABA: a plain `fetch_add` on an *integer* is immune to ABA, because the value's meaning does not change when it recycles. ABA only bites the *CAS-loop* form of read-modify-write on a *pointer*, where you compare-and-swap a value whose identity matters. So when you can express your update as a `fetch_add`/`fetch_or` rather than a CAS loop over a recycled pointer, do — it dodges the entire ABA conversation.

Python deserves a footnote here, because its story is special: under the Global Interpreter Lock, a single bytecode is atomic but `x += 1` compiles to multiple bytecodes (LOAD, ADD, STORE) and so *still* loses updates across threads — the GIL does not save you. The fix is the same (a `Lock` or an atomic structure), and the free-threaded future changes the calculus. That whole story belongs to the python-performance series; see [the GIL explained: what it protects and what it costs](/blog/software-development/python-performance/the-gil-explained-what-it-protects-and-what-it-costs) rather than me re-deriving it here.

## The fixes, organized

When you have met all four hazards and their cures, the cures organize into a small tree, and seeing the structure is what lets you recall the right fix under pressure instead of guessing.

![A tree grouping the defenses into two branches where making the operation atomic covers double width compare and swap atomic types and locks while versioning the value covers tagged pointers and hazard pointers or epochs](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-7.png)

Every fix in this post is one of two moves. **Make the operation atomic** — collapse the multi-step operation into one indivisible step the hardware or runtime serializes: a double-width CAS, an `atomic<T>`, a lock, an `O_CREAT|O_EXCL`, a `computeIfAbsent`. **Or version the value** — attach something that changes even when the bits recycle, so a stale reference becomes detectable: a tagged pointer's counter, a hazard pointer's announcement, an epoch's grace period. The tree figure above lays this out: two roots, the concrete techniques hanging under each. ABA can be fixed either way (tag = version it; reclamation = version it more thoroughly; though if you also need to *dereference* the recycled node safely, you need reclamation, not just a tag). TOCTOU, torn access, and lost updates are all "make it atomic." Keep the two-branch shape in your head and the specific fix follows.

And here is the practical map from "what does my code look like" to "which hazard and which fix" — the checklist I actually run when reviewing concurrent code:

![A matrix mapping code patterns which are lock free compare and swap, a file or resource check, a wide value, and lazy initialization to the hazard each invites and the mitigation that closes it](/imgs/blogs/the-aba-problem-toctou-and-torn-reads-8.png)

| Code pattern | Hazard it invites | Mitigation |
| --- | --- | --- |
| CAS loop over a recycled pointer | ABA | tagged pointer, hazard pointer, DWCAS |
| `access`/`stat` then `open`/`exec` | TOCTOU | open to a fd, `O_NOFOLLOW`, `openat`, `O_EXCL` |
| read/write a 64-bit value or struct unsynchronized | torn read/write | atomic of T, alignment, lock, immutable swap |
| lazy `if (x == null) x = new()` | broken DCL / publish-before-construct | `volatile`/release, `call_once`, `sync.Once` |

The last matrix figure carries these rows. If you internalize one artifact from this post, make it this table: it turns four scary, rare, timing-dependent bugs into four recognizable code smells with four standard cures.

## Measured: reproducing the ABA bug honestly

These bugs are rare and timing-dependent, and honesty about *how* rare is part of understanding them. Let me describe how I reproduce ABA and what the numbers actually look like — with the caveat that every number here is platform- and scheduler-dependent and should be read as an order of magnitude, not a precise constant. I am describing the *method* so you can run it yourself; I am not quoting a benchmark you should cite.

The setup: a Treiber stack with the broken plain-pointer pop, populated with a few nodes, hammered by two kinds of threads. "Poppers" pop a node, do a tiny bit of work, and `push` it back (immediately recycling addresses — this is what makes ABA likely). A "victim" thread runs the broken pop with a deliberately widened window: between reading `head`/`head.next` and the CAS, it yields or spins briefly, giving the poppers time to do an A → B → A cycle. A checker validates structural invariants after each round — that the node count is conserved, that no node appears twice, that `head` reaches a known sentinel by following `next`.

What you observe, qualitatively and reliably:

- **With the broken pop and a widened window**, the invariant check fails — duplicated nodes, lost nodes, or a cycle in the list — within seconds to minutes on a multi-core box. The failure rate is highly sensitive to the artificial delay: a few microseconds of yield in the window can take the per-operation corruption probability from "essentially never in a test run" to "several failures per second." That sensitivity is the whole lesson: **the bug's frequency is a function of the window width and the scheduler, not a fixed property of the code.** In the field, with no artificial delay, the window is a handful of instructions and you might see it once a day under heavy load — exactly the maddening cadence from my opening story.
- **With the tagged-pointer pop** (same workload, same widened window), the invariant check passes indefinitely. You can confirm the mechanism is actually firing by counting CAS retries: the tagged version shows *more* retries (the stale-tag CASes that correctly fail and loop) but zero invariant violations. The plain version shows *fewer* retries — because the dangerous CASes wrongly *succeed* instead of retrying — and accumulating corruption. More failed-and-retried CASes is the *signature of correctness* here, which is counterintuitive until you see why.
- **Without the artificial delay at all**, you may run the broken stack for a long time and see nothing, then get a single corruption under a burst of contention. This is the trap: a clean test run does not mean the code is correct, it means the window did not happen to be hit. ABA cannot be reliably caught by testing; it must be reasoned away.

Reproducing a torn read is its own small craft, and the method is instructive. Pin a writer thread to one core and a reader thread to another (so they truly run in parallel, not time-sliced on one core). The writer alternates a wide value between two patterns whose halves are easy to tell apart — say `0x0000000000000000` and `0xFFFFFFFFFFFFFFFF` — in a tight loop with no synchronization. The reader loads the value in a tight loop and checks: did I ever see a value that is *neither* all-zeros nor all-ones? On a 64-bit target where the access is naturally aligned and the platform guarantees 64-bit atomicity, you will *never* see a mixed value, and that is the point — it confirms the hardware guarantee. Run the same harness with a *misaligned* 64-bit value straddling a cache line, or a two-field struct, and the mixed values (`0x00000000FFFFFFFF`, `0xFFFFFFFF00000000`) start appearing — proof the access tore. The rate again depends entirely on the platform, the alignment, and how hard the two cores contend; it is a demonstration, not a benchmark to quote. The discipline is the same as for ABA: pin threads, busy-loop to widen the window, run long, and treat the *presence* of an impossible value as the signal — its frequency is noise you do not control.

The honest measurement discipline that matters: warm up the allocator and threads before measuring; run many rounds because a single run is meaningless for a probabilistic bug; report the *conditions* (core count, allocator, the artificial window) alongside any rate, because they dominate the number; and never present a single reproduction as "the" failure rate — it is one sample from a distribution whose mean you do not control. The same discipline applies to TOCTOU (an attacker measures the win rate over thousands of trials, not one) and to torn reads (you tighten the race by pinning the writer and reader to different cores and busy-looping, then check for an impossible value). If you take one methodological rule from this post: **for rare concurrency bugs, the absence of a failure in testing is evidence of nothing.** Reason about the window; do not trust the dice.

## Case studies / real-world

These hazards are not academic. Each has a famous, documented incident.

**The Java double-checked locking saga.** Before Java 5, the broken DCL idiom above was published in books and recommended widely — and it was *broken on real JVMs*, not just in theory, because the pre-JSR-133 memory model genuinely permitted the constructor-publish reorder. The community wrote "The 'Double-Checked Locking is Broken' Declaration," signed by a long list of concurrency experts including Doug Lea and David Holmes, documenting precisely why the idiom could not be fixed within the old model. The resolution came with JSR-133 (Java 5, 2004), which redefined `volatile` to carry acquire-release semantics, making the `volatile`-field version correct. This is the canonical example of a memory-model bug that *no amount of locking fixes* — only an ordering guarantee does. It is also why "just make it `volatile`" became folklore: in this exact idiom, it is the precise and necessary fix. Brian Goetz's *Java Concurrency in Practice* recommends the initialization-on-demand holder idiom (a static nested class) as the cleaner alternative that sidesteps DCL entirely.

**TOCTOU in real CVEs.** The `access()`/`open()` race is not a toy; it has produced concrete local privilege escalations across decades. A well-known representative is the historical class of setuid TOCTOU bugs catalogued under CWE-367 (Time-of-check Time-of-use Race Condition); the `mktemp`/`tmpnam` family of "predictable temp file" races (CWE-377) is its close cousin, where a program checks a temp path then creates it non-atomically and an attacker pre-creates a symlink. The defensive migration the whole industry made — from `access` + `open` to `open` with `O_NOFOLLOW`/`O_EXCL`, from `tmpnam` to `mkstemp`, from path-based to `*at`-based APIs — is a direct response to this hazard class. More recently, file-system race conditions remain a recurring theme in container-escape and sandbox-bypass research, where an attacker swaps a path between a sandbox's validation and the kernel's use of it. The pattern never went away; the APIs got safer.

**The Linux kernel and ABA / reclamation.** The kernel's pervasive use of **RCU** is in large part a response to exactly the safe-reclamation problem that underlies ABA: how do you let many readers traverse a structure lock-free while a writer removes and frees nodes, without a reader dereferencing freed memory or hitting an ABA on a recycled node? RCU's answer — defer the free until a grace period passes in which every reader has been quiescent — is one of the most successful concurrency primitives in any production system, carrying an enormous fraction of the kernel's read-mostly data structures. It is the industrial-strength version of "version the value / defer the free" from our taxonomy, and it is exactly why the dedicated [memory reclamation post](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu) exists. The lock-free literature (Maged Michael's hazard-pointer paper; the IBM/Treiber-stack lineage) grew up around demonstrating ABA in real allocators and proving these schemes correct.

Where I have given a number-free description, that is deliberate: the precise CVE identifiers, JVM versions, and kernel timelines are worth looking up at their primary sources rather than trusting a remembered figure, and the *mechanism* — not the version number — is what transfers.

## When to reach for this (and when not to)

A decisive section, because every one of these fixes is a cost and some are large.

**Worry about ABA only when you are writing a lock-free structure with a CAS loop over a recycled pointer.** That is a narrow situation. If you are using a mutex, a channel, or a standard concurrent collection from your language's library, ABA is the library author's problem, already solved. **Do not hand-roll a lock-free stack to avoid a mutex you have not measured as a bottleneck** — the lock-free version is dramatically harder to get right (ABA, reclamation, the memory model) and frequently *not faster* under realistic contention. Reach for tagged pointers or hazard pointers only after profiling proves the lock is the bottleneck and the structure is genuinely hot. When you do go lock-free, prefer a battle-tested library (crossbeam, folly, the JDK's `Atomic*` and `Concurrent*` classes) over your own CAS loop.

**Worry about TOCTOU whenever a security or correctness decision depends on a name you re-resolve** — file paths in privileged code above all, but also any check-then-act on shared state across threads. The fix is cheap (operate on a handle, use the atomic compound primitive) and the failure is catastrophic (root, data loss), so the cost/benefit is lopsided: just write it the safe way by default. In non-privileged, single-threaded code where no adversary or concurrent writer exists, TOCTOU cannot occur — do not contort the code defending against a race that is structurally impossible.

**Worry about torn reads/writes whenever you share a value wider than a machine word without synchronization** — 64-bit values on 32-bit targets, structs, fat pointers, anything multi-word. The fix is nearly free (declare it atomic, or keep the shared value word-sized) and the bug is a data race (UB in C/C++), so again: default to the safe form. Do *not*, however, slap an atomic on every field reflexively — atomics have a cost on the contended path, and a value that is only ever touched by one thread, or only under a lock you already hold, needs nothing extra. Atomicity is for *shared mutable* values; the word "shared" is load-bearing.

**Worry about broken DCL whenever you lazily initialize shared state.** The right move is almost always to *not* hand-roll it: use `std::call_once`, a function-local `static`, `sync.Once`, `OnceLock`, or an eager `final`/`static` initialization if the cost of always-constructing is acceptable. Hand-rolled DCL is a code smell in 2026 — the standard library primitive is correct, readable, and impossible to get the barrier wrong. Reach for explicit acquire/release atomics only when you have a genuinely custom publication pattern the library primitives do not cover, and then prove the ordering on paper.

The meta-rule across all four: **these are precisely the bugs where "it passed the tests" means nothing.** They are rare, timing- and platform-dependent, and invisible to most race detectors when the operation is technically atomic. The defense is not more testing; it is recognizing the *shape* — a value compared but recycled, a name checked then re-resolved, a value wider than the access, a pointer published before construction — and applying the standard cure on sight.

## Key takeaways

1. **CAS verifies a value, not a version.** A pointer that goes A → B → A reads as unchanged, so a stale CAS succeeds and corrupts. That is the ABA problem, and it is invisible to data-race detectors because every access is atomic.
2. **Fix ABA by versioning or by deferring the free.** A tagged pointer (ABA counter, swapped via double-width CAS) makes a recycled A compare unequal; hazard pointers, epochs, and RCU make the node unrecyclable while anyone holds it — the stronger fix, because it also prevents use-after-free.
3. **TOCTOU is ABA at the OS level.** Checking a name (`access`) then acting on it (`open`) trusts a binding that can change in the window. Operate on a handle (a fd), use an atomic check-and-act (`O_EXCL`, `computeIfAbsent`), or operate-then-handle-failure.
4. **Torn reads/writes are a width problem, not a logic problem.** A value wider than the atomic unit, or one that straddles a cache line, can be read half-old and half-new — a value that never existed. Use an atomic type, keep shared values word-sized and aligned, or lock the wide struct.
5. **Plain wide values shared across threads are a data race** — undefined behavior in C/C++, and historically tearable for non-`volatile` `long`/`double` in Java. "It's one variable" is not "it's atomic"; atomicity is about access width and alignment.
6. **Double-checked locking needs a barrier, not just a lock.** The lock provides mutual exclusion; it does *not* order the constructor's writes before the pointer's publish. Without an acquire-release edge (`volatile` in Java, explicit memory orders in C++), a reader can see a non-null pointer to a half-built object.
7. **Don't hand-roll lazy init or lock-free structures.** Use `sync.Once`, `std::call_once`, `OnceLock`, function-local statics, and library concurrent collections. They bake in the ordering you would otherwise get wrong on a weakly-ordered CPU.
8. **Every fix is one of two moves: make the operation atomic, or version the value.** Collapse the multi-step operation into one indivisible step, or attach something that changes when the bits recycle. Knowing which move a hazard needs is the whole skill.
9. **For rare concurrency bugs, a clean test run is evidence of nothing.** The window simply did not get hit. Reason the bug away from the structure of the code; do not trust the scheduler's dice.

## Further reading

- *The Art of Multiprocessor Programming*, Herlihy & Shavit — the rigorous treatment of CAS, the ABA problem, lock-free stacks and queues, and the progress hierarchy.
- *C++ Concurrency in Action*, Anthony Williams — practical `std::atomic`, memory orders, lock-free data structures, and the DWCAS / tagged-pointer mechanics in real C++.
- *Java Concurrency in Practice*, Brian Goetz et al. — the JMM, `volatile` semantics, safe publication, and the holder-class idiom that supersedes hand-rolled DCL.
- "The 'Double-Checked Locking is Broken' Declaration" and the JSR-133 (Java Memory Model) documents — why DCL was broken before Java 5 and what `volatile` was redefined to guarantee.
- Maged Michael, "Hazard Pointers: Safe Memory Reclamation for Lock-Free Objects" — the foundational paper on reclamation and the ABA-without-recycling cure.
- CWE-367 (Time-of-check Time-of-use) and CWE-377 (insecure temporary file) — the catalogued TOCTOU hazard classes with the standard mitigations.
- Within this series: [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), [shared mutable state and the anatomy of a race condition](/blog/software-development/concurrency/shared-mutable-state-and-the-anatomy-of-a-race-condition), [compare-and-swap and building lock-free data structures](/blog/software-development/concurrency/compare-and-swap-and-building-lock-free-data-structures), [memory reclamation: hazard pointers, epochs, and RCU](/blog/software-development/concurrency/memory-reclamation-hazard-pointers-epochs-and-rcu), [memory barriers: acquire, release, and fences](/blog/software-development/concurrency/memory-barriers-acquire-release-and-fences), and the capstone [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
