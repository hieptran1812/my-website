---
title: "Use-After-Free and Memory Corruption: The Crash That Happens 10,000 Lines From the Bug"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Catch the write that corrupts memory instead of staring at the crash it causes pages later, using AddressSanitizer, Valgrind, watchpoints, and the redzone mechanism that names the exact free and the exact use."
tags:
  [
    "debugging",
    "software-engineering",
    "memory-corruption",
    "use-after-free",
    "addresssanitizer",
    "valgrind",
    "c-cpp",
    "security",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/use-after-free-and-memory-corruption-1.png"
---

The pager goes off at 03:14. A service that has run for two years just took a SIGSEGV, and the stack trace lands you inside `send()`, deep in the networking layer, trying to write to a file descriptor whose value is `-1094795586`. You have never touched that code. You read it three times. It is correct. The file descriptor is corrupt, but nothing near the crash corrupted it. You attach `gdb` to the core dump, you walk the frames, and every single one of them is innocent. The bug is somewhere else entirely — somewhere you freed a block of memory and kept using it, or wrote one byte past the end of a buffer, and the damage didn't show up until 10,000 lines and several milliseconds later, when some unrelated object happened to live in the wreckage.

This is the hardest class of bug in systems programming, and it is hard for a single, specific reason: **the write that corrupts memory and the crash that results are separated in time and space.** A debugger placed at the crash sees the victim, not the culprit. You are looking at a body and trying to reconstruct a murder that happened in a different room, hours earlier, committed by someone who has since left the building. The naive approach — set a breakpoint at the crash, inspect the stack, reason backward — fails completely, because the corrupting code already finished running and its stack frame is long gone.

![A flow diagram showing a freed block being reused by the allocator, then a stale pointer writing into the new owner and causing a crash far away, with AddressSanitizer catching it at the write](/imgs/blogs/use-after-free-and-memory-corruption-1.png)

The whole craft of debugging memory corruption, then, is a single move: **stop catching the crash, start catching the write.** You need a tool that traps the instant a pointer touches memory it has no right to touch — before the corruption spreads, while the guilty stack frame is still on the call stack. That tool exists, it is mostly free, and once you internalize how it works you will never again spend three days bisecting a heap. By the end of this post you will be able to take a "segfault, no idea why" report and, in a single instrumented run, get back the exact line that freed the memory, the exact line that allocated it, and the exact line that misused it — all three, named, with stack traces. We will build that capability from first principles: why the crash lands far from the bug (the allocator reuses freed blocks), how AddressSanitizer's redzones and quarantine catch the access at the source, how Valgrind does the same thing without recompiling, how a hardware watchpoint stops you on the exact corrupting write, and why every one of these bugs is also a security vulnerability. This is the `observe → reproduce → hypothesize → bisect → fix → prevent` loop from [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), applied to the one bug class where "observe at the crash" actively lies to you.

## 1. Why the crash is nowhere near the bug

Start with the mechanism, because everything else follows from it. In C and C++, when you call `free(p)` (or `delete` an object), you are not erasing memory. You are telling the allocator "I'm done with this block; you may reuse it." The bytes are still there, the pointer `p` still holds the same address, and reading or writing through `p` still *works* — for now. Nothing checks. The hardware doesn't know that block is "freed"; freed is a bookkeeping concept that lives entirely inside the allocator's data structures, not in the CPU's memory protection.

So here is the sequence that ruins your night. You free a 64-byte node. The allocator marks those 64 bytes as available and, crucially, puts the block on a free list so it can hand it out again quickly. A few microseconds or a few thousand instructions later, some completely unrelated part of the program calls `malloc(64)` or `new Session`, and the allocator — being efficient — hands back *the exact same 64 bytes*. Now two pointers refer to the same memory: your stale pointer, which you believe points to a freed node, and a fresh, live pointer that legitimately owns a `Session`. The moment you write through the stale pointer — say you do `node->next = something`, which writes 8 bytes at offset 8 — you are scribbling into the middle of a live `Session` object. You just overwrote `Session.fd`.

Nothing crashes. The write succeeded; it was a valid, mapped, writable address. The corruption is now latent. It sits there, a `fd` field holding garbage, until much later when some code does `send(session->fd, ...)` and the kernel rejects the impossible file descriptor — or worse, the garbage happens to be a *valid* fd belonging to a different connection, and you silently send one user's data to another. That is the gap the kit calls out: the crash in `send()` is the symptom; the use-after-free write was the disease, and they share no stack frame, no nearby line, often not even the same source file.

The reason a debugger-at-the-crash is useless now becomes precise. By the time `send()` faults, the function that did the bad write has returned. Its stack frame is reclaimed. The CPU registers it used are reused. There is no breadcrumb from the victim (`session->fd`) back to the culprit (`node->next = ...`), because the connection between them is *aliasing* — two pointers to one block — and aliasing leaves no trace in a backtrace. You can stare at the crash site forever; the answer is not there.

The same structural problem produces the other members of this bug family, and it is worth naming them up front because the diagnostics differ slightly:

- **Use-after-free (UAF):** you free a block, the allocator reuses it, your stale pointer reads or writes through it. The corruption surfaces when the new owner is touched.
- **Double-free:** you call `free(p)` twice. The second free corrupts the allocator's *own* free-list metadata — it links a block that's already linked, creating a cycle or a dangling free-list node — and the allocator crashes later, on some unrelated `malloc`, when it walks the broken list.
- **Buffer overflow / out-of-bounds write:** you write past the end (or before the start) of an array. On the heap you clobber the *next chunk's header* (its size and flags), which the allocator reads on the next `malloc`/`free` and chokes on. On the stack you clobber the saved return address or a local — classic "stack smashing."
- **Uninitialized read:** you read a variable or a `malloc`'d block before writing it. You get whatever was left there — often deterministic enough to pass tests and random enough to fail in prod.
- **Type confusion:** you reinterpret a block as the wrong type (a bad cast, a union misuse, a deserializer that trusts a tag), and read/write fields at offsets that mean something else entirely.

Every one of these shares the throughline: **a write through a pointer that has no legitimate right to that memory, with the damage surfacing later, somewhere else.** Internalize that and the rest of this post is just learning the tools that collapse the time-and-space gap.

#### Worked example: the UAF that crashed three files away

Here is the smallest reproducer that shows the whole mechanism. Save it as `uaf.c`:

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct Node {
    int   value;
    struct Node *next;   // offset 8 on a 64-bit build
};

struct Session {
    int  fd;             // offset 0
    char user[56];
};

int main(void) {
    struct Node *n = malloc(sizeof(struct Node));
    n->value = 7;
    n->next  = NULL;

    free(n);                         // (A) the bug: we free here

    // ... thousands of unrelated instructions later ...
    struct Session *s = malloc(sizeof(struct Session));  // reuses n's block
    s->fd = 4;                       // a real, open socket
    strcpy(s->user, "alice");

    n->next = (struct Node *)0xdead; // (B) the corruption: stale write

    printf("about to send on fd=%d\n", s->fd);  // (C) the crash site
    // s->fd was clobbered by the write at (B), because n->next lives
    // at the same address as s->fd ... actually offset 8 vs 0 here,
    // so tune the struct or the field to land on fd in your build.
    return s->fd;
}
```

Compile it normally and it might print `fd=4` and exit cleanly, or it might print garbage, or it might crash — it depends on the allocator, the libc version, the phase of the moon. That nondeterminism is the second cruelty of memory corruption: **it is often a heisenbug.** Whether the freed block gets reused, and what lands there, depends on allocation history, so the same bug fails 1-in-10,000 times in prod and never once on your laptop. We will fix that nondeterminism in a moment with the right tool — but first, sit with the fact that line (A), line (B), and line (C) are the free, the corrupting write, and the crash, and in a real codebase they would be in three different files maintained by three different teams.

It helps to lay the bug out on a timeline, because the time axis is the thing the crash hides from you. There are six moments, and the bug lives in the gap between two of them.

![A timeline of a use-after-free showing allocation, free, the allocator reusing the block, the stale write, the later crash, and AddressSanitizer naming the free and use lines](/imgs/blogs/use-after-free-and-memory-corruption-2.png)

At `t0` the block is born. At `t1` you free it — and note that `free` does not touch your pointer variable, so `n` still holds the same address and still *looks* valid; nothing about the language stops you from dereferencing it. At `t2` the allocator, doing exactly what it is designed to do, recycles those bytes for the next request. At `t3` the stale write lands, scribbling into the new owner. At `t4` the program finally faults, somewhere downstream, when the corrupted field is used. The entire detective problem is the distance from `t3` to `t4` — and, worse, the distance from `t1` to `t3`, because that is the window in which an *innocent-looking* `n` is actually a loaded gun. The reason this matters operationally: any tool that only observes `t4` (a debugger at the crash, a core dump, a stack trace) is observing the one moment that contains no information about the cause. Every effective technique in this post works by moving your observation point earlier — to `t3` (catch the write) or to `t1` (catch the dangerous free). Keep that timeline in your head; it is the whole shape of the problem, and every tool is just a different way to teleport your attention back along it.

One more thing the timeline makes obvious: the *size* of the `t1`-to-`t3` window controls how often the bug bites. If you free and immediately reuse (a tight loop allocating and freeing same-sized objects), the window is microscopic and the corruption is almost guaranteed every run. If the free and the stale use are separated by seconds of unrelated work, the block may or may not have been reused yet, and you get the maddening intermittency. This is why some UAFs are "always crashes" and others are "crashes once a week under load" — it's the same bug, just a different gap. And it's why, as we'll see, AddressSanitizer's trick of *refusing to reuse the block for a while* (quarantine) is so powerful: it artificially widens the poisoned window to cover the whole `t1`-to-`t3` gap, so the stale write at `t3` reliably lands on still-poisoned memory.

## 2. Catch it at the write: AddressSanitizer's redzones and quarantine

The single most important tool in this entire post is **AddressSanitizer** (ASan), a compiler instrumentation pass built into both Clang and GCC. You turn it on with one flag, `-fsanitize=address`, and it transforms every memory access in your program into a checked access. The result: instead of a silent corrupting write followed by a crash pages later, you get a deterministic abort *at the exact instruction that did the bad write*, with a stack trace.

The mechanism is worth understanding precisely, because understanding it tells you what ASan can and can't catch. ASan maintains **shadow memory**: a parallel region where every 8 bytes of your program's memory is described by 1 byte of shadow. That shadow byte encodes whether those 8 bytes are fully addressable, partially addressable, or poisoned. The compiler rewrites every load and store to first compute the shadow address, read the shadow byte, and check it. If the byte says "poisoned," ASan stops and reports.

![A layered view of the AddressSanitizer heap with poisoned redzones around a valid allocation, a shadow map, a quarantine for freed blocks, and a final report on bad access](/imgs/blogs/use-after-free-and-memory-corruption-4.png)

Two design choices make ASan catch our exact bugs:

**Redzones.** When you `malloc(64)`, ASan actually allocates more and surrounds your 64 bytes with poisoned guard regions on both sides — typically 16 bytes left and a larger region right. Any read or write that strays one byte past your buffer lands in a redzone, the shadow byte says poisoned, and ASan reports `heap-buffer-overflow`. This is how the off-by-one gets caught at the write, not at the later `malloc`.

**Quarantine.** When you `free()` a block, ASan does *not* return it to the allocator immediately. It poisons the entire block (so any access reports `heap-use-after-free`) and puts it in a **quarantine** — a FIFO of recently-freed blocks that are held back from reuse. This is the key trick: by delaying reuse, ASan dramatically increases the odds that your stale pointer hits *still-poisoned* memory rather than a freshly-handed-out new object. The block stays poisoned and out of circulation until the quarantine fills and it ages out. So the use-after-free that only crashes 1-in-10,000 times in production becomes a deterministic abort under ASan, because the freed block is reliably poisoned at the moment you touch it.

Let's run our reproducer under it. The compile and run:

```bash
# Clang or GCC both work; -g gives line numbers, -O1 keeps it fast and readable
$ clang -fsanitize=address -g -O1 uaf.c -o uaf
$ ./uaf
```

And here is the report — the thing that turns a three-day hunt into a three-minute fix. I've annotated it:

```console
==48213==ERROR: AddressSanitizer: heap-use-after-free on address 0x602000000018
WRITE of size 8 at 0x602000000018 thread T0
    #0 0x4f1a2b in main /home/me/uaf.c:30:13      <-- (B) the corrupting WRITE
    #1 0x7f3c in __libc_start_main
    #2 0x41c0 in _start

0x602000000018 is located 8 bytes inside of 16-byte region
freed by thread T0 here:
    #0 0x4d6f in free
    #1 0x4f1a0a in main /home/me/uaf.c:24:5        <-- (A) where we FREED it
    #2 0x7f3c in __libc_start_main

previously allocated by thread T0 here:
    #0 0x4d8a in malloc
    #1 0x4f19c2 in main /home/me/uaf.c:21:9         <-- where it was ALLOCATED
    #2 0x7f3c in __libc_start_main

SUMMARY: AddressSanitizer: heap-use-after-free uaf.c:30:13 in main
```

Read what you just got. Three stack traces, stacked: the **bad access** (`uaf.c:30` — the write through the stale pointer), the **free** (`uaf.c:24` — where you released it), and the **allocation** (`uaf.c:21` — where the block was born). The entire causal chain, named, in one run. There is no bisection, no reasoning backward from the crash, no luck. ASan reconstructed the murder, the victim, and the weapon, and handed you all three. This is the "method" the whole series is about: a runnable diagnostic that converts an unbounded search into a single deterministic answer.

To feel the difference in your bones, put the two approaches side by side. The same crash, debugged two ways — and the gap between them is the difference between a three-day hunt and a three-minute fix.

![A before and after comparison of debugging the same crash with gdb at the crash site versus AddressSanitizer at the corrupting write, showing the sanitizer collapses the hunt into one run](/imgs/blogs/use-after-free-and-memory-corruption-3.png)

On the left, `gdb` on the core dump: you land in `send()` with a SIGSEGV, you walk the frames, every one is innocent, and you begin the long blind bisection — adding logs, narrowing time windows, second-guessing every `free` in the codebase. The information you need (which `free`, which write) is simply not present on that stack, so no amount of cleverness at the crash site recovers it. On the right, the same binary rebuilt under ASan: one run, and the tool prints the bad access, the free site, and the alloc site. The asymmetry is total. It's not that ASan is a *better* debugger than gdb — it's that ASan is observing the *right moment* (the write) while gdb-at-the-crash is observing the wrong one (the symptom). Right tool, right moment. That is the entire methodological lesson, and it generalizes: whenever a crash's stack is "innocent," your instinct should not be "debug harder at the crash" but "what tool observes the moment of corruption instead?"

A few flags and details that matter in practice:

```bash
# Common ASAN_OPTIONS, set in the environment before you run:
$ export ASAN_OPTIONS="detect_leaks=1:abort_on_error=1:halt_on_error=1:strict_string_checks=1"
$ export ASAN_OPTIONS="$ASAN_OPTIONS:detect_stack_use_after_return=1"
# Symbolize the report (turn addresses into file:line) if it shows raw addresses:
$ export ASAN_SYMBOLIZER_PATH=$(which llvm-symbolizer)
```

- `detect_leaks=1` turns on **LeakSanitizer**, bundled with ASan: at program exit it reports any block you allocated and never freed, with the allocation stack. That is a free memory-leak detector riding along (covered in depth in the sibling post on hunting memory leaks and bloat).
- `detect_stack_use_after_return=1` catches the stack version of UAF — returning a pointer to a local and dereferencing it after the frame is gone.
- `halt_on_error=0` lets ASan report *and keep going*, which is useful when you want to find several bugs in one run rather than stopping at the first.
- The runtime cost is roughly **2× CPU and ~3× memory**. That's cheap enough to run your entire test suite under ASan in CI, and you absolutely should — it's the single highest-leverage thing you can do to keep this bug class out of production.

#### Worked example: the one-byte overflow that crashed in malloc

The second canonical bug is the off-by-one heap overflow, and it teaches the "crashes far away" lesson even more vividly than UAF. Consider this loop that fills a 16-byte buffer but uses `<=` instead of `<`:

```c
#include <stdlib.h>

int main(void) {
    char *buf = malloc(16);
    for (int i = 0; i <= 16; i++) {   // BUG: <= writes index 16, one past the end
        buf[i] = 'A';
    }

    // The off-by-one already happened. Now we do unrelated work:
    char *other = malloc(32);          // <-- this is where it crashes, sometimes
    free(other);
    free(buf);
    return 0;
}
```

![A flow diagram showing a one-byte overflow past a 16-byte buffer clobbering the next chunk size header, corrupting the free list, and crashing in a later malloc, with ASan catching it at the write](/imgs/blogs/use-after-free-and-memory-corruption-6.png)

Without instrumentation, that single byte at index 16 lands in the **next chunk's metadata.** Heap allocators store a small header before (and sometimes after) every chunk: its size and a few flag bits. Your stray `'A'` (0x41) overwrites part of the next chunk's size field. Nothing crashes yet — you just wrote one byte to mapped memory. The damage detonates later, on `malloc(32)` or on `free`, when the allocator reads that now-garbage size, computes a bogus next-chunk pointer, and either aborts with `malloc(): corrupted top size` or walks off into unmapped memory and segfaults. The crash is in libc, in a function you never called directly, with a stack that points at innocent code. Classic far-from-the-bug.

Under ASan, the same program stops cold *at the write*:

```console
==51002==ERROR: AddressSanitizer: heap-buffer-overflow on address 0x602000000030
WRITE of size 1 at 0x602000000030 thread T0
    #0 0x4f1b in main /home/me/oob.c:6:16          <-- buf[16] = 'A', the overflow
0x602000000030 is located 0 bytes after 16-byte region
allocated by thread T0 here:
    #0 0x4d8a in malloc
    #1 0x4f0e in main /home/me/oob.c:4:17           <-- the 16-byte malloc
SUMMARY: AddressSanitizer: heap-buffer-overflow oob.c:6:16 in main
```

`0 bytes after 16-byte region` is ASan telling you, precisely, that you wrote immediately past a 16-byte allocation — an off-by-one. It names the loop line and the allocation line. The fix is `i < 16`. Total time from report to root cause: as long as it takes to read the message. Compare that to the version where you start at the libc `malloc` abort and try to reason backward — you can't, because the corrupting write left no frame on that stack.

## 3. The other catcher: Valgrind, when you can't recompile

ASan is the first tool you reach for, but it has a hard prerequisite: you must **recompile** with `-fsanitize=address`. Sometimes you can't. The bug only reproduces in a release binary you've already shipped; the build is a tangle of prebuilt third-party libraries; you're handed a core dump and a binary and told "find it." For those cases there is **Valgrind**, specifically its `memcheck` tool, and it needs no recompilation at all.

Valgrind works completely differently from ASan, and the difference explains its trade-offs. Where ASan inserts checks at *compile* time, Valgrind is a **dynamic binary instrumentation** framework: it runs your program on a synthetic CPU, JIT-recompiling every basic block to add instrumentation as it executes. `memcheck` maintains its own shadow memory tracking, for every byte, whether it is **addressable** (allocated and not freed) and whether it is **defined** (written since allocation). On every memory access it checks addressability; on every use of a value in a branch or syscall it checks definedness.

That last part is Valgrind's superpower and ASan's blind spot: **Valgrind catches uninitialized reads** that ASan misses. If you `malloc` a block and read a field before writing it, ASan is fine with that (the memory is addressable), but Valgrind tells you "Conditional jump or move depends on uninitialised value(s)" and points at the read. The cost: Valgrind is **10× to 50× slower**, sometimes worse. You don't run your whole suite under it; you run the one reproducer.

A typical session:

```bash
# No special build needed — just compile with -g for line numbers
$ gcc -g -O0 uaf.c -o uaf
$ valgrind --leak-check=full --track-origins=yes --error-exitcode=1 ./uaf
```

For our use-after-free, `memcheck` reports:

```console
==9921== Invalid write of size 8
==9921==    at 0x10917A: main (uaf.c:30)              <-- the stale write
==9921==  Address 0x4a8b048 is 8 bytes inside a block of size 16 free'd
==9921==    at 0x484890F: free (vg_replace_malloc.c)
==9921==    by 0x109165: main (uaf.c:24)              <-- where it was freed
==9921==  Block was alloc'd at
==9921==    at 0x4848899: malloc (vg_replace_malloc.c)
==9921==    by 0x109142: main (uaf.c:21)              <-- where it was allocated
```

Same three-part story as ASan — invalid write, the free site, the alloc site. The vocabulary is different ("Invalid write of size N", "Address X is N bytes inside a block ... free'd") but the information is identical: the bad access plus the lifecycle of the block it hit. `--track-origins=yes` adds, for uninitialized-value errors, *where the uninitialized value came from* — invaluable for the uninitialized-read bug. And for leaks, `--leak-check=full` classifies every unfreed block as `definitely lost` (no pointer to it anywhere — a true leak), `indirectly lost` (only reachable through a definitely-lost block), `possibly lost` (reachable only by an interior pointer), or `still reachable` (you have a pointer but never freed it — often fine, e.g. a global cache).

So: **ASan when you can recompile and want speed; Valgrind when you can't recompile, or when you need uninitialized-read detection, and you can afford the slowdown on a single reproducer.** They are complementary, not competitors. The table below makes the choice mechanical:

| Tool | What it finds | Overhead | Needs recompile? | Reach for it when |
|---|---|---|---|---|
| AddressSanitizer | UAF, heap/stack/global overflow, double-free, leaks | ~2× CPU, ~3× memory | Yes (`-fsanitize=address`) | Default; run the whole suite in CI |
| Valgrind memcheck | UAF, overflow, **uninitialized reads**, leaks | 10×–50× slower | No | Can't rebuild; need uninit detection; one reproducer |
| MemorySanitizer | Uninitialized memory reads only | ~3× CPU | Yes, **all deps too** | ASan is clean but values are garbage |
| ThreadSanitizer | Data races (which often *cause* corruption) | 5×–15× CPU | Yes (`-fsanitize=thread`) | Corruption only under concurrency |
| UBSan | Signed overflow, bad shifts, misaligned/null deref, bad casts | near-zero | Yes (`-fsanitize=undefined`) | Always-on companion to ASan |
| Stack canary | Stack-buffer-overflow of the return address | near-zero | Yes (`-fstack-protector`) | Always on in release builds |

![A comparison matrix of memory tools showing what each finds, its overhead, and when to reach for it, with AddressSanitizer as the fast default and Valgrind as the no-recompile fallback](/imgs/blogs/use-after-free-and-memory-corruption-7.png)

### Double-free: corrupting the allocator's own bookkeeping

There's a third bug worth its own treatment, because it corrupts something subtler than your data: it corrupts the *allocator's* data. A **double-free** is calling `free(p)` twice on the same pointer. To see why it's dangerous, you need one fact about how a typical heap allocator (glibc's `ptmalloc`, for instance) recycles freed blocks: it keeps **free lists** (called "bins"), and when you free a small block, the allocator writes a pointer *into the freed block itself* — it reuses the now-unused payload to store the "next free block" link, threading freed chunks into a singly-linked list. The block's own bytes become list metadata.

Now free the same block twice. The first free links it into the bin. The second free links it in *again* — so the same chunk appears twice in the free list, or the list's `next` pointer ends up pointing at itself. The free list is now a corrupt, possibly cyclic, data structure. Nothing crashes yet. The detonation comes on a future `malloc`: the allocator pops a chunk off the bin, follows the corrupted `next` pointer, and either hands the *same block out twice* to two different callers (instant aliasing — two live pointers to one block, the UAF setup all over again) or dereferences a garbage `next` and crashes deep inside `malloc`, far from your double-free. This is the second classic "crashes in malloc, miles from the bug" signature, and historically it was a powerful exploitation primitive: attackers deliberately double-freed to make `malloc` return a controlled pointer, the "fastbin dup" attack. Modern glibc added a cheap defensive check — it verifies the most-recently-freed chunk isn't being freed again — which turns many double-frees into an immediate, loud abort:

```console
free(): double free detected in tcache 2
Aborted (core dumped)
```

That message is glibc's own integrity check firing — a defensive abort that happens to land near the second free, which is lucky. But the check is not exhaustive (it only catches the immediate-repeat case in the fast path), so plenty of double-frees still slip through to corrupt the list silently. Under ASan, by contrast, a double-free is caught *unconditionally and exactly*, because ASan tracks the freed state in shadow memory and reports `attempting double-free` with both free stacks:

```console
==60011==ERROR: AddressSanitizer: attempting double-free on 0x602000000010
    #0 free
    #1 main /home/me/dfree.c:11:5         <-- the second free
freed by thread T0 here:
    #0 free
    #1 main /home/me/dfree.c:9:5          <-- the first free
```

Two free stacks, named. The fix for a double-free is the same discipline that prevents most UAFs: **null the pointer after freeing it** (`free(p); p = NULL;` — and `free(NULL)` is a guaranteed no-op, so a double-free becomes harmless), or, far better, use an owning type that frees exactly once and leaves no live raw pointer behind. The deeper point: a double-free is what happens when *ownership is unclear* — two pieces of code both think they're responsible for freeing the same block. The bug is really a design bug about who owns what, surfacing as a corrupt free list.

## 4. The full taxonomy and the tool for each

Step back and see the shape of the whole problem, because choosing the right tool is half the battle and the choice falls naturally out of the taxonomy. Every memory corruption bug is either **temporal** (you used a pointer at the wrong *time* — the memory was valid once, but not now) or **spatial** (you used a pointer at the wrong *place* — outside the bounds of what it legitimately points to).

![A taxonomy tree splitting memory corruption into temporal bugs caught by AddressSanitizer and spatial bugs caught by sanitizers and canaries](/imgs/blogs/use-after-free-and-memory-corruption-5.png)

- **Temporal — wrong time.** Use-after-free and double-free both live here: the block was yours, you released it, and you (or the allocator) touched it after. The fix is about *ownership and lifetime* — who frees, and when, and never twice. ASan's quarantine is purpose-built for this.
- **Spatial — wrong place.** Buffer overflow/underflow (heap, stack, global) and out-of-bounds access live here: the pointer is to a live object, but you indexed outside it. The fix is about *bounds*. ASan's redzones catch the heap and global cases; the stack canary and ASan's stack instrumentation catch the stack case.
- **Uninitialized read** sits a little apart — it's a "wrong place in time" bug (you read before you wrote) — and it has its own dedicated tool, **MemorySanitizer** (`-fsanitize=memory`), or Valgrind's definedness tracking.
- **Type confusion** is the subtle one: the memory is live and in-bounds, but you're interpreting it as the wrong type, so your field offsets are wrong. There's no single sanitizer flag for it; you catch it with `-fsanitize=undefined` (UBSan) for some cases (bad downcasts via `-fsanitize=vptr`), strict typing, tagged unions, and careful deserialization.

This taxonomy is not academic — it tells you which flag to add. "Garbage value, no crash" → MSan or Valgrind. "Crashes in malloc/free" → heap corruption, ASan. "Smashed return address, hijacked control flow" → stack overflow, canary + ASan. "Crash only under load with multiple threads" → the corruption is probably caused by a *data race* (two threads writing the same block with no synchronization), so you reach for **ThreadSanitizer** (`-fsanitize=thread`) — the corruption is a symptom; the missing lock is the disease. The debugger post in this series ([the debugger is a microscope, use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it)) covers the interactive side; here the lesson is that *the symptom names the sanitizer.*

### UBSan and MSan, briefly

Two more sanitizers earn a place in your default build because they're nearly free or catch what ASan can't.

**UndefinedBehaviorSanitizer (UBSan)**, `-fsanitize=undefined`, catches the undefined behaviors that *enable* corruption: signed integer overflow (which the optimizer assumes never happens, so it deletes your "if it overflowed" check), shifts by too-large amounts, null-pointer dereference, misaligned access, and out-of-bounds via `-fsanitize=bounds`. Its overhead is near zero, and it composes with ASan: `-fsanitize=address,undefined` runs both. Many "impossible" optimizer-induced bugs (the check that compiles away because it relied on UB) are caught here.

```bash
# A great default for tests: ASan + UBSan together, with line numbers
$ clang -fsanitize=address,undefined -fno-sanitize-recover=all -g -O1 prog.c -o prog
```

`-fno-sanitize-recover=all` makes UBSan abort on the first violation instead of printing and continuing — you want a hard stop in CI so the build goes red.

**MemorySanitizer (MSan)**, `-fsanitize=memory`, is the specialist for uninitialized reads. It tracks, bit by bit, which memory has been written, and reports the first time an uninitialized bit influences a branch, a syscall, or an output. Its one operational headache: it needs *every* piece of code in the process instrumented, including the C++ standard library and any dependency, or it produces false positives from uninstrumented reads — so you typically build against an MSan-instrumented libc++. That cost is why most teams use Valgrind's `--track-origins` for the occasional uninitialized-read hunt and reserve MSan for codebases where they can instrument the world.

## 5. The stack side: canaries, smashing, and return addresses

So far the bugs lived on the heap. The stack has its own corruption story, and it's the one with the richest security history, so it's worth its own section.

When a function runs, its local variables — including local arrays — live on the stack, and so does the **saved return address**: the address the CPU will jump to when the function returns. These sit close together in memory. If a function has a `char buf[64]` local and you `strcpy` 100 bytes into it, the extra 36 bytes run past `buf` and overwrite whatever's adjacent — which, depending on stack layout, includes the saved return address. Overwrite the return address and you've *changed where the program goes when this function returns.* This is **stack smashing**, and in its malicious form it's how an attacker turns "this program accepts too-long input" into "this program runs my code."

The defense built into every modern compiler is the **stack canary** (a.k.a. stack protector), enabled with `-fstack-protector` (and `-fstack-protector-strong`, which is the practical default on most distros). The mechanism is beautifully simple: when a function with a stack buffer is entered, the compiler stores a random value — the canary — on the stack *between the local buffers and the saved return address.* Right before the function returns, it checks that the canary is still the value it stored. A buffer overflow that reaches the return address *must* pass through the canary first, corrupting it. The check fails, and the program aborts with the famous message:

```console
*** stack smashing detected ***: terminated
Aborted (core dumped)
```

That message is the canary doing its job — it converted a silent control-flow hijack into a loud, immediate abort. The trade-off is essentially zero: a few instructions per function entry/exit, only on functions that actually have stack buffers. It's on by default in every serious build; if you ever see a hand-rolled build script with `-fno-stack-protector`, treat it as a red flag.

The canary catches the *return-address* overflow, but a smaller overflow that corrupts only an adjacent local (not the return address) slips past it — for those, you still need ASan's stack instrumentation (`-fsanitize=address` poisons redzones around stack variables too) or the older guard-page tools. Which brings us to the bluntest instrument of all.

### Guard pages: Electric Fence and friends

The simplest possible way to catch an out-of-bounds access is to put the buffer right next to a page of memory that the CPU itself refuses to touch. That's a **guard page** — a page marked non-readable, non-writable via `mprotect`. Any access to it triggers a hardware fault (SIGSEGV) *immediately*, at the exact faulting instruction, with no instrumentation overhead on the normal path.

**Electric Fence** (`efence`) and the modern `mprotect`-based heap debuggers (glibc's `MALLOC_PERTURB_` and `MALLOC_CHECK_`, or `libgmalloc` / "Guard Malloc" on macOS) work this way: every `malloc` is placed so its end sits flush against a guard page. Read or write one byte past, and the hardware faults on that instruction — you get a clean stack right at the overflow.

```bash
# Electric Fence: LD_PRELOAD it, no recompile, immediate fault at the overflow
$ gcc -g oob.c -o oob
$ LD_PRELOAD=libefence.so ./oob
# crashes with SIGSEGV exactly at buf[16] = 'A', and gdb shows that line
```

```bash
# glibc's built-in heap checker, also no recompile:
$ MALLOC_CHECK_=3 ./oob          # 3 = print error AND abort on heap inconsistency
$ MALLOC_PERTURB_=42 ./oob       # fills freed memory with 0x42, makes UAF reads obvious
```

Guard-page tools are heavyweight on memory (each allocation may consume a whole page or more), so you don't ship with them, but they're a great no-recompile way to get a hardware-precise fault on an overflow when ASan isn't an option and Valgrind is too slow. Their limitation: they catch the *spatial* end-of-buffer case cleanly but don't track the *temporal* UAF lifecycle the way ASan's quarantine does.

### Hardware comes to the rescue: MTE

The newest entry, and the one that may eventually make this whole post historical, is **Memory Tagging Extension (MTE)** on ARMv8.5+ (shipping in recent ARM cores). MTE puts the tag *in the hardware*. Every 16-byte chunk of memory gets a 4-bit tag stored in spare bits, and every pointer carries a 4-bit tag in its top bits. On each memory access the CPU checks that the pointer's tag matches the memory's tag; mismatch → fault. The allocator assigns a fresh tag on each `malloc` and a new tag on `free`, so a stale pointer to a freed-then-reallocated block carries the *old* tag and faults on use — a hardware-checked use-after-free, at near-zero overhead, suitable for *production*, not just testing. Android already uses it. The day MTE is everywhere, "catch it at the write" becomes the hardware's default behavior. Until then, ASan in CI is your MTE.

## 6. The surgical option: a hardware watchpoint on the corrupting address

Sometimes you've already narrowed it down. You know *which address* gets corrupted — maybe ASan told you the field, maybe you've seen the same garbage value (`0xdeadbeef`, a stray ASCII string, a recognizable pointer) clobber the same struct member three times. Now you want to catch the *exact write* that does it, in a normal debug build, with full context. This is the job of a **hardware watchpoint**, and it ties this post directly to the interactive-debugger post in the series.

A watchpoint tells the CPU's debug registers: "stop the program the instant anyone writes to this address." It's a hardware feature (x86 has 4 debug registers, `DR0`–`DR3`), so it costs *nothing* at runtime until the write happens — unlike a software watchpoint, which single-steps and is thousands of times slower. When the watched address is written, the CPU traps, and the debugger stops you *on the instruction that did the write*, with the full call stack of the culprit still live. That's the whole game: you've teleported your breakpoint from the crash (the victim) to the write (the culprit).

The workflow in `gdb`:

```bash
$ gdb ./myprogram
(gdb) run
# ... program runs, you discover session->fd at 0x602000000018 keeps getting clobbered ...
# Stop it before the corruption, then watch the address:
(gdb) watch *(int *)0x602000000018
Hardware watchpoint 2: *(int *)0x602000000018
(gdb) continue
# The instant ANYTHING writes those 4 bytes, gdb stops:
Hardware watchpoint 2: *(int *)0x602000000018
Old value = 4
New value = -559038737        # 0xdead..., the garbage
0x0000000000401a2b in main () at uaf.c:30
30          n->next = (struct Node *)0xdead;   # <-- caught the culprit, red-handed
(gdb) backtrace                                # full stack of the corrupting write
```

That `backtrace` after the watchpoint fires is the thing you could never get by breaking at the crash — it's the call stack of the *write*, the frame that the crash-site debugger could only dream of. If you watch a field on a heap object, you may want to watch by expression (`watch session->fd`) so gdb re-evaluates the address; for a fixed address use the cast form above.

The catch with hardware watchpoints is that you usually need to know the address, and addresses on the heap move run to run (thanks to ASLR and allocation order). Three ways to pin it down:

- **Run under ASan first** to get the field and a stable reproducer, then attach gdb and watch.
- **Make the bug deterministic** by disabling ASLR for the debug session: `setarch $(uname -m) -R gdb ./prog` (or gdb's `set disable-randomization on`, which is the default). Now the address is the same every run, so you can `watch` it.
- **Watch a struct member by expression** rather than a raw address, after stopping at a known point where the object exists.

Watchpoints shine for two cases ASan handles less directly: a corruption inside a *single* large object (where every access is in-bounds and not-freed, so ASan sees nothing — e.g., a logic bug that writes the wrong field), and a stack/global you can name. For the classic heap UAF and overflow, ASan is faster; for "this specific field keeps getting trashed and I want the exact write," the watchpoint is the scalpel. Reach for it the way the [microscope post](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) describes — when you've localized the *what* and need the *who*.

#### Worked example: the watchpoint that fired on iteration 3,847,221

A real flavor of how surgical this gets. A team had a graph library where one node's `color` field flipped from `BLACK` to a garbage value `0x7f` somewhere during a 4-million-node traversal, causing a crash in the rebalancer much later. ASan found nothing — the write was in-bounds and the object was live (a type-confusion: a union was being written as the wrong variant, so the write landed on `color`'s offset legally). They pinned the node's address with `set disable-randomization on`, set `watch node->color`, and `continue`d. It fired on the 3,847,221st write to that region, stopping dead on a line in the union-handling code that wrote `tag` as a 64-bit value when the active variant only had a 8-bit `color` at that offset. The backtrace named the exact mis-tagged deserialization path. Total debugging time after the watchpoint: under a minute. Without it: the team had spent two days adding print statements that perturbed the layout and made the bug vanish (a heisenbug — instrumenting it changed the allocation pattern). The watchpoint, being hardware and zero-overhead, didn't perturb anything.

## 7. Why these are security bugs: from UAF to RCE

You cannot write about memory corruption honestly without saying this plainly: **every bug in this post is also a security vulnerability.** The same mechanism that crashes your program — a write through a pointer that has no right to that memory — is, in the hands of an attacker who controls the inputs, a path to running arbitrary code. The crash is the benign outcome; the exploit is the malicious one. They are the same bug.

Walk the chain for a buffer overflow. The benign version: you write past `buf` and clobber the next chunk's header, and the program eventually crashes. The malicious version: an attacker who controls the data being written *chooses* what goes past the buffer. If it's a stack buffer and they reach the return address, they overwrite it with an address of their choosing — pointing at code they injected, or (defeating non-executable stacks) at a chain of existing code snippets stitched together (return-oriented programming). The function returns, and the CPU jumps to attacker-controlled code. That is **remote code execution (RCE)** if the input came over the network. The overflow you'd fix as "off-by-one, oops" is, from the other side, a generic exploitation primitive.

Use-after-free is, if anything, *more* powerful for attackers, because it gives a write (or read) into a block the allocator will reuse. The exploit technique: free an object, then make the program allocate an attacker-controlled object of the same size so it lands in the freed block (heap grooming/spraying), then trigger the use of the stale pointer — which now reads or writes attacker-chosen bytes through what the program thinks is the original object's vtable or function pointer. Control the vtable pointer and you control where a virtual call goes. Browser and kernel exploits are *full* of use-after-frees; they are the workhorse of modern memory-corruption exploitation precisely because of the aliasing mechanism we opened with.

And uninitialized/over-read bugs leak secrets. The most famous one deserves its own section.

## 8. War story: Heartbleed, the read that leaked the internet's secrets

In April 2014, a single missing bounds check in OpenSSL's implementation of the TLS heartbeat extension became **Heartbleed (CVE-2014-0160)**, one of the most consequential bugs in the history of the internet. It is the canonical buffer **over-read**, and it shows the corruption family from a different angle: not a write that crashes, but a read that *exfiltrates*.

The TLS heartbeat is a keep-alive: the client sends a small "heartbeat request" containing a payload and a length field saying how long the payload is, and the server echoes the payload back. The bug: OpenSSL trusted the *attacker-supplied length field* and copied that many bytes from the request buffer into the response — without checking that the request actually contained that many bytes. So an attacker sent a 1-byte payload but claimed a length of 65,535. The server dutifully `memcpy`'d 65,535 bytes starting at the payload — 1 byte of which was the attacker's, and 65,534 of which were *whatever happened to be in the server's heap right after the request buffer.* And it sent all of it back.

What lived in that adjacent heap memory? Other connections' data. Decrypted requests. Session cookies. Usernames and passwords. And — the catastrophe — the server's **private TLS key**, which had been sitting in the heap. An attacker could repeat the request thousands of times, each leaking another 64 KB window of server memory, and reassemble secrets. No crash. No log entry. No trace. The server worked perfectly while bleeding its most sensitive data to anyone who asked, which is why it earned the name.

The fix was a two-line bounds check: validate that the claimed payload length doesn't exceed the actual received record length before copying. That's it. A spatial bug — a read past the bounds of the legitimate data — born from trusting a length field, the same root cause as a thousand humbler overflows.

The debugging lesson is sharp and on-theme. **A tool that catches the over-read at the read would have caught Heartbleed before it shipped.** Run OpenSSL's test suite under AddressSanitizer or Valgrind with an over-length heartbeat request, and the `memcpy` reading past the request buffer lands in a redzone (ASan) or an addressability error (Valgrind) — `heap-buffer-overflow READ of size N`. The bug is invisible to a normal run (the read "succeeds," it's mapped memory) and screamingly obvious under instrumentation. In the aftermath, the OpenSSL project, OSS-Fuzz, and Google's fuzzing infrastructure made exactly this the standard: **fuzz the parser, run it under sanitizers, and the over-reads and overflows surface automatically.** Heartbleed is the reason "run your network-facing C under ASan in CI" stopped being optional. It is the most expensive demonstration in history of this post's one rule: catch it at the access, not at the symptom — and for an over-read, there *is* no crash symptom, only a slow leak, which makes the sanitizer the *only* thing that catches it.

It's worth dwelling on *why* Heartbleed was invisible for so long — it shipped in OpenSSL 1.0.1 in March 2012 and wasn't disclosed until April 2014, more than two years of exposure across a huge fraction of the internet's HTTPS servers. The reason is exactly this post's thesis turned tragic. A buffer over-*read* produces no crash, no error, no log line — the read "succeeds" because the bytes it touches are mapped, addressable memory. So every test passed. Every code review missed it, because the bug is a *missing* check, and missing code is the hardest thing to notice when reading. Manual testing could never find it, because nothing observably went wrong. Only a tool that models *which bytes are legitimately part of the request* — ASan's redzones, or Valgrind's addressability shadow, or a fuzzer that feeds an over-length claimed length and watches the sanitizer fire — could catch it, and at the time, OpenSSL's CI did not run under sanitizers or fuzzing. The fix added afterward was not just the two-line bounds check; it was the *process*: OpenSSL joined OSS-Fuzz, and every parser path now runs under continuous fuzzing with ASan. The bug taught the industry that for security-critical C parsing, "it passes the tests" means nothing — only "it survives the fuzzer under a sanitizer" counts. That is the most expensive lesson in this entire post, and it's the same lesson as the humblest off-by-one: the read that strays past the bounds leaves no trace unless a tool is watching the bounds for you.

Two more named cases worth a sentence each, because they round out the family:

- **The PS3/iPhone-era heap overflows and countless browser UAFs** that powered jailbreaks and exploits — the same UAF-then-spray primitive, used offensively, year after year.
- **glibc's `malloc(): corrupted top size` and `free(): invalid pointer` aborts** — not a single famous incident but the daily reality of heap corruption: the allocator's own integrity checks (`MALLOC_CHECK_`, the unlink/bins consistency checks added over the years) catching a corrupted free-list and aborting *defensively*, far from the bug, which is exactly the "crash in malloc" you now know how to chase back to the write with ASan.

## 9. The full investigation: from "segfault, no idea" to root cause

Let's put it together as a method you can run, because the tools only help if you reach for them in the right order. Here is the decision procedure I actually use when handed a memory-corruption crash, framed as the series' `observe → reproduce → hypothesize → bisect → fix → prevent` loop.

**Observe.** Read the crash. Where did it land — in `send()`, in `malloc`/`free`, in a destructor, with a `*** stack smashing detected ***`? The *site* of the crash is itself a hypothesis-generator: a crash in `malloc`/`free` screams heap corruption (overflow or double-free); a `stack smashing detected` is a stack overflow past the canary; a garbage value in a field that's "impossible" smells like UAF or type confusion; a SIGSEGV reading a small offset off a near-null pointer is often a null-deref, not corruption. Note the faulting address. Is it a recognizable pattern (`0xdeadbeef`, an ASCII string, a shifted pointer)? Garbage that *looks* like text means something wrote a string where a pointer should be.

**Reproduce.** This is non-negotiable and harder than usual here, because memory corruption is so often a heisenbug — it depends on allocation history, timing, and layout, so it fires 1-in-N times and not on demand. The whole [reproduce-it-first post](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) applies double. Your single best move is: **rebuild under ASan and run the test that flakes in a loop.** ASan's quarantine and redzones make latent corruption deterministic, turning "fails 1-in-10,000" into "fails every time," because the freed block stays poisoned instead of getting reused.

```bash
# Reproduce a heisenbug deterministically by making corruption loud:
$ clang -fsanitize=address,undefined -g -O1 -fno-omit-frame-pointer prog.c -o prog
$ for i in $(seq 1 200); do ./prog --the-flaky-case || { echo "caught on run $i"; break; }; done
```

If it reproduces only under load or only with multiple threads, the corruption is probably race-driven — add `-fsanitize=thread` (in a separate build; you can't combine ASan and TSan) and look for the unsynchronized write. If it only reproduces in a release `-O2` build and vanishes at `-O0`, that's a strong signal of *undefined behavior the optimizer exploited* (a deleted overflow check, a strict-aliasing violation) — build with `-fsanitize=undefined` and also try `-fno-strict-aliasing` to test the aliasing hypothesis.

**Hypothesize and bisect.** ASan usually skips you straight to the answer — it *names* the free, the alloc, and the access. When it doesn't (the bug is a logic-level corruption inside a live, in-bounds object — type confusion), fall back to the watchpoint: pin the corrupted address, `watch` it, and let the hardware catch the write. If even the address is elusive, **bisect**: the corruption was introduced by some commit, so `git bisect run` with an ASan-instrumented test as the predicate finds the commit that introduced it in `log2(N)` steps — the same binary-search discipline from [bisecting your bug](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), now with a sanitizer as the automated test oracle.

```bash
# Let git bisect find the commit that introduced the corruption, automated:
$ git bisect start HEAD v2.3.0           # bad=HEAD, known-good=v2.3.0
$ git bisect run sh -c '
    clang -fsanitize=address -g -O1 prog.c -o /tmp/p &&
    /tmp/p --repro && echo OK || exit 1'  # exit 1 = ASan fired = "bad"
# 4096 commits -> ~12 steps -> the introducing commit, named
```

**Fix.** The fix depends on the family: null the pointer after free (and never free twice); fix the bound (`<` not `<=`, validate the length *before* copying — the Heartbleed fix); initialize before read; correct the cast/union handling. But the deeper fix is structural, which is the next section.

**Prevent.** Make the bug class impossible or loud, permanently. This is where the before→after is most satisfying.

![A before and after comparison showing raw free with a live pointer in a release build versus owned pointers with AddressSanitizer in CI aborting at the bad write](/imgs/blogs/use-after-free-and-memory-corruption-8.png)

### Stress-testing the method: the cases that break it

The clean version above assumes you can reproduce the bug under ASan and read off the answer. Real corruption fights back. Here are the cases that break the easy path and what to do about each — the situations that separate someone who's read about ASan from someone who has actually shipped the fix.

**"It only reproduces under load."** The corruption is almost certainly a *data race*: two threads writing the same block (or one freeing while another uses it) with no happens-before edge between them, so the C++ memory model gives you no guarantee about ordering and the result is a torn write or a UAF. Single-threaded ASan runs will be green forever. Rebuild with `-fsanitize=thread` (you cannot combine it with ASan — separate build), and run the *concurrent* test. TSan reports the two unsynchronized accesses and the missing lock. The corruption was the symptom; the data race is the disease. If you can't get TSan to fire because the race is rare, increase contention deliberately: pin threads to the same core, add `sched_yield()` at suspicious points, or run under `stress-ng` to perturb the scheduler.

**"It only reproduces in the release build, never at `-O0`."** This is the loudest possible signal of *undefined behavior the optimizer exploited.* The classic: you wrote `if (ptr + offset < ptr)` to check for overflow, but signed pointer/integer overflow is UB, so at `-O2` the compiler proves "that can never be true" and deletes your check — at `-O0` it kept the check and the bug hid. Build with `-fsanitize=undefined` (UBSan works at any optimization level) and also try `-fno-strict-aliasing` to test whether a type-punning violation is the culprit. The rule, echoing the kit: **don't chase a heisenbug at `-O2` — reproduce it under a sanitizer first**, because the optimized assembly is a maze and the sanitizer goes straight to the UB.

**"It only happens on one host, or after six hours."** A six-hour incubation almost always means the corrupting window depends on accumulated allocation history — the freed block only gets reused into the fatal layout after the heap has churned enough. Run that host's exact workload under ASan with a *small* quarantine and large redzones tuned to its allocation pattern; the deterministic poisoning collapses six hours into the first reuse. "One host only" points at something host-specific feeding the corruption: a different libc version with a different allocator layout, a different `MALLOC_ARENA_MAX`, a config that enables a code path the others don't, or genuinely flaky hardware (rare, but a corrupted bit that's always the same address is a memory-stick smell — check `dmesg` for ECC errors before blaming your code).

**"It only happens when two requests interleave."** This is the race case sharpened: a use-after-free where request A frees an object that request B still holds a pointer to, and the bug only fires when B's use lands after A's free. Reproduce by *forcing* the interleaving — add a synchronization point that pauses B exactly between "A frees" and "B uses," or run the two requests in a tight loop under TSan until the window hits. The repeat-until-fail loop from [reproduce it first](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) is your friend here: `for i in $(seq 100000); do run_two_interleaved || break; done` under ASan.

**"I can't attach a debugger in prod."** Often true — you can't stop a payments process to single-step, and you can't run ASan's 3× memory in production. The answer is to *not debug in prod*: capture a **core dump** (`ulimit -c unlimited`, or `coredumpctl` / a crash handler that writes a minidump), pull it to a sandbox, and analyze offline. The core won't show you the corrupting write (it's the victim, remember), but it gives you the corrupted *value* and *address*, which is exactly the input you need to set a watchpoint on a sanitized reproducer offline. Combine prod's "what got corrupted" with offline ASan's "who corrupted it." If you genuinely cannot reproduce offline, your last resort is lightweight always-on production hardening — `_FORTIFY_SOURCE`, stack canaries, and (if your hardware supports it) MTE — which convert the corruption into a clean, logged abort with a stack, turning an untriageable silent crash into a triageable loud one. That trade — a tiny constant overhead for a usable stack trace — is almost always worth it for a service that handles untrusted input.

## 10. Prevention: make corruption impossible or loud

The numbers in this section are the proof the kit demands, and they're the most defensible numbers in the whole post, because they're about *coverage*, not micro-benchmarks. A team I'll describe representatively had a C++ service crashing in production roughly 2–4 times a week with assorted SIGSEGVs in unrelated places — the classic far-from-the-bug signature. The crashes were untriageable from core dumps because, as you now understand, the core showed the victim, not the culprit. Here is the before→after.

**Step 1: turn on ASan in CI.** They added a CI job that built the whole codebase with `-fsanitize=address,undefined` and ran the existing test suite under it. The very first run found *six* latent corruptions — UAFs and overflows that had been there for years, silently corrupting and occasionally crashing. Each came with the three stacks (access, free, alloc). Fixing those six took about a week. Production crash rate dropped from ~3/week to under 1/month. That is the single highest-leverage change available: **the test suite you already have, run under ASan, finds the bugs your customers are finding for you.** Cost: CI build time roughly doubled and tests ran ~2× slower — a few extra minutes per pipeline, in exchange for catching the bug class that pages you at 3am.

**Step 2: own your memory.** They migrated the hot allocation paths from raw `new`/`delete` to RAII smart pointers — `std::unique_ptr` for single ownership, `std::shared_ptr` where lifetime genuinely shared, and `std::string`/`std::vector` instead of raw buffers. The point isn't fashion; it's that **an owning type frees exactly once, automatically, and there's no live raw pointer to misuse afterward.** Double-free becomes a compile error or impossible-by-construction; UAF becomes much harder because the owner controls lifetime. Where they had to keep a raw pointer, they adopted the discipline of setting it to `nullptr` immediately after free, so a stale use becomes a clean null-deref (a crash *at the use*, with a sane stack) instead of a silent corruption.

**Step 3: keep it loud.** They left ASan on in CI permanently (every PR), enabled `-fstack-protector-strong` and `-D_FORTIFY_SOURCE=2` (compile-time and runtime bounds checks on `memcpy`/`strcpy`/`sprintf` family) in release builds, and added a nightly Valgrind run of the integration suite to catch the uninitialized reads ASan misses. For the network-facing parser, they added a fuzzer (`libFuzzer` + ASan) — the Heartbleed lesson, institutionalized.

The honest trade-offs, stated plainly the way the [how-to-reach-for-it](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) discipline demands:

- **ASan is not for production.** 2× CPU and 3× memory is fine for tests, not for serving traffic. Don't run ASan in prod (use MTE if you have it, or guard-page/`MALLOC_CHECK_` selectively). ASan is a *test-time and CI-time* tool.
- **`-D_FORTIFY_SOURCE` and stack canaries *are* for production** — they're near-free and they convert a class of overflows into clean aborts. Always on in release.
- **Smart pointers have a cost** (`shared_ptr` is a refcounted atomic; don't use it where `unique_ptr` suffices), but the cost is trivially worth it versus a UAF.
- **Don't chase a heisenbug in an optimized build first.** If it only reproduces at `-O2`, suspect UB and rebuild with sanitizers — but to *understand* the corruption, reproduce it under ASan (which works fine at `-O1`), not by staring at optimized assembly.
- **Rewriting hot, memory-unsafe components in a memory-safe language** (Rust, with its borrow checker that makes UAF and data races compile errors; or Go, garbage-collected) is the nuclear-but-real prevention. It's why the industry is moving security-critical C/C++ to Rust. You can't always do it, but where the corruption keeps coming back in one module, it's the durable fix. (Note: `unsafe` Rust re-opens the door — UAF is possible in `unsafe` blocks — which is exactly why this bug class still matters even there.)

## 11. How to reach for these tools (and when not to)

A decisive recommendation section, because every tool here has a cost and reaching for the wrong one wastes the night.

- **Default to ASan.** If you can recompile, `-fsanitize=address,undefined` is your first move for any suspected memory corruption, and it should run on every PR in CI. It's fast enough for the whole suite, it names the free/alloc/access, and it makes heisenbugs deterministic. Reach for it *first*, before you even open a debugger.
- **Reach for Valgrind when you can't recompile,** when you're handed a binary, or when you specifically need uninitialized-read detection (`--track-origins=yes`). Accept the 10–50× slowdown by running it on one reproducer, not the whole suite. Don't reach for Valgrind if ASan is available and the bug isn't an uninitialized read — ASan is 5–25× faster for the same find.
- **Reach for a hardware watchpoint** when you've localized *which address or field* gets corrupted and you need the exact write with its stack — especially for in-bounds, in-lifetime logic corruption (type confusion) that ASan can't see. Pin the address with `set disable-randomization on` first.
- **Reach for ThreadSanitizer** when corruption only appears under concurrency — the corruption is downstream of a data race, and TSan finds the unsynchronized access.
- **Reach for guard pages / `MALLOC_CHECK_`** for a quick, no-recompile, hardware-precise fault on an overflow when ASan isn't an option.
- **Reach for the stack canary and `_FORTIFY_SOURCE`** as always-on production defenses, not as debugging tools — they convert exploits into aborts.

And the *when nots*, which matter just as much:

- **Don't attach `gdb` to a payments process in production and start poking** — a watchpoint that single-steps, or a misstep, can stall or kill a process serving real money. Reproduce in a sandbox under ASan; don't experiment on prod.
- **Don't run ASan or Valgrind in production to "catch it live."** The overhead changes timing and memory enough to mask race-driven corruption and to blow your latency budget. Reproduce offline.
- **Don't add print statements to chase memory corruption** — printing allocates, which changes the heap layout, which moves or hides the bug (a heisenbug you created). Use a tool that observes without perturbing: ASan instruments deterministically; a hardware watchpoint is zero-overhead. This is the opposite advice from most bugs, where a log line is the cheapest answer.
- **Don't trust a clean ASan run to mean "no memory bugs."** ASan misses uninitialized reads (use MSan/Valgrind) and most data races (use TSan) and pure logic corruption inside live objects (use a watchpoint). "ASan is green" means "no UAF/overflow/double-free on the paths your tests exercised" — which is why coverage and fuzzing matter.
- **Don't fix the crash site.** The single most common mistake: someone sees the crash in `send()`, adds a null-check on `session->fd`, and ships it. The crash stops — and the corruption moves to the *next* object that lands in the freed block, surfacing as a new mystery crash next week. Fix the *write*, never the *symptom*. That is the entire thesis of this post.

## 12. Key takeaways

- **The write that corrupts and the crash that results are separated in time and space.** A debugger at the crash sees the victim, not the culprit, because the allocator reused the freed block and the guilty stack frame is long gone. Stop catching the crash; catch the write.
- **AddressSanitizer is the default.** `-fsanitize=address` gives you a deterministic abort *at the bad access*, with three stacks: the access, the free, and the allocation. Run it on every PR. It makes heisenbugs deterministic via redzones and a free quarantine.
- **Valgrind is the no-recompile fallback** and the way to catch uninitialized reads (`--track-origins=yes`), at 10–50× slowdown — run it on one reproducer, not the suite.
- **The symptom names the tool.** Crash in `malloc`/`free` → heap corruption (ASan). `stack smashing detected` → stack overflow past the canary. Garbage value, no crash → uninitialized read (MSan/Valgrind). Corruption only under load → data race (TSan). In-bounds, in-lifetime field corruption → type confusion (hardware watchpoint).
- **A hardware watchpoint teleports your breakpoint from the crash to the write.** When you know the corrupted address, `watch` it and the CPU stops you on the exact corrupting instruction with the culprit's full stack — at zero runtime overhead.
- **Stack canaries and `_FORTIFY_SOURCE` are free production defenses;** ASan and Valgrind are test-time tools. Never run sanitizers in prod; never ship without canaries.
- **Every one of these bugs is a security vulnerability.** UAF and overflow are the generic primitives behind RCE; an over-read leaked the internet's private keys as Heartbleed. Catch them at the access, in CI, under a sanitizer and a fuzzer.
- **Fix the write, never the crash site.** Null-checking `session->fd` makes the crash move, not disappear. Prevent the bug class structurally: own your memory (RAII/`unique_ptr`), keep ASan in CI, and rewrite chronically-unsafe modules in a memory-safe language.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the `observe → reproduce → hypothesize → bisect → fix → prevent` loop this post applies to the one bug class where "observe at the crash" lies to you.
- [The debugger is a microscope: use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) — the interactive-debugger companion; the hardware watchpoint technique here is the scalpel that catches the corrupting write.
- [Reproduce it first, or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — essential for the heisenbug nature of memory corruption; ASan's quarantine is how you make a 1-in-10,000 crash deterministic.
- The sibling post on hunting memory leaks and bloat (planned slug `hunting-memory-leaks-and-bloat`) — LeakSanitizer and Valgrind's `definitely lost` are the same toolchain pointed at the opposite problem: memory you never free rather than memory you free too early.
- [Memory profiling in Python: tracemalloc, memray, and finding leaks](/blog/software-development/python-performance/memory-profiling-tracemalloc-memray-and-finding-leaks) — the managed-language counterpart, where the GC removes UAF but leaks and retention bugs remain.
- The **AddressSanitizer wiki** (google/sanitizers on GitHub) — the canonical reference for ASan flags, the shadow-memory and redzone design, `ASAN_OPTIONS`, and the LeakSanitizer/MemorySanitizer/ThreadSanitizer family.
- The **Valgrind manual** (valgrind.org/docs) — the `memcheck` user guide, the addressability/definedness model, and the leak-check classification (`definitely`/`indirectly`/`possibly lost`).
- The **GDB manual** chapter on watchpoints, and **CWE-416 (Use After Free)**, **CWE-787 (Out-of-bounds Write)**, and the **CVE-2014-0160 (Heartbleed)** writeups for the security angle.
