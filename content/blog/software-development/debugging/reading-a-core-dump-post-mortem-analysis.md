---
title: "Reading a Core Dump: Post-Mortem Analysis of a Crash You Never Saw"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn to capture a core dump reliably from any runtime or container, symbolize a stripped binary, and perform the autopsy in gdb, dlv, MAT, and dotnet-dump so a crash that never reproduces becomes a one-line root cause."
tags:
  [
    "debugging",
    "software-engineering",
    "core-dump",
    "post-mortem",
    "gdb",
    "symbolization",
    "heap-dump",
    "crash-triage",
    "root-cause-analysis",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/reading-a-core-dump-post-mortem-analysis-1.png"
---

The crash came in over the weekend, and nobody was watching. Monday morning there is one line in the dashboard: a payments worker exited with signal 11 at 02:47 on Saturday, restarted automatically, and has been fine ever since. You try to reproduce it. You replay the same request a thousand times. Green every time. You run it under load. Green. You point the fuzzer at it for an hour. Green. The bug happened once, in production, on a host you do not control anymore, and it left no log line worth reading — just `segmentation fault (core dumped)` and a process that died in 200 microseconds. You did not have a debugger attached. You will never have a debugger attached, because the bug is a one-in-ten-million interleaving you cannot recreate. By every normal rule of debugging, you are stuck.

Except that the kernel did something for you at the moment of death. The phrase `(core dumped)` is not a tombstone — it is a gift. When the process took that fatal signal, the kernel froze it. It wrote out the entire memory image, every register, every thread's stack, the heap, the locals, and the exact faulting instruction, all to a file on disk. That file is a **core dump**, and it is the frozen corpse of your process at the instant it died. Post-mortem debugging is the autopsy: you open that corpse in a debugger and inspect the dead process *exactly as if you had been attached at the moment it crashed*. You do not need to reproduce anything. The evidence is already on the table. You just have to know how to read it.

![A branching flow showing a fatal SIGSEGV signal, the kernel snapshotting memory and registers into a frozen core file, the symbolize step that matches a build-id to the debug file, and gdb walking the core to a one-pointer root cause](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-1.png)

That figure is the whole arc of this post in one picture. A fatal signal triggers the kernel to snapshot the process into a core file; if your binary is stripped the frames come back as `?? ()` until you feed it matching debug symbols; then `gdb prog core` lets you walk the dead stack to the exact line that killed it. By the end of this post you will be able to do six concrete things. You will **make cores exist reliably** — `ulimit -c unlimited`, the `core_pattern` and `coredumpctl` plumbing, `gcore` to snapshot a *live* process without killing it, and the per-runtime switches (`GOTRACEBACK=crash`, `-XX:+HeapDumpOnOutOfMemoryError`, `dotnet-dump collect`, Python's `faulthandler`). You will **capture cores from containers and from the field**, where the `core_pattern` is host-wide and the crash happened on someone else's laptop. You will **symbolize a stripped binary**, understanding exactly why it shows `?? ()` and how a `build-id` ties a core to the one binary that produced it. You will **read the core** in `gdb` with `bt`, `thread apply all bt`, `frame`, `info locals`, `info registers`, and `x/`, and do the equivalent autopsy with `dlv core`, Eclipse MAT, and `dotnet-dump analyze`. You will **triage crashes at scale**, bucketing ten thousand reports into five bugs by stack signature. And you will do all of this on a crash you never saw and cannot reproduce. This is the deepest application of the series' master loop — observe, reproduce, hypothesize, bisect, fix, prevent — because a core dump lets you *observe* perfectly even when *reproduce* is impossible. If you have not read it, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) sets up that loop; this post is what you do when the bug refuses to come back and the only witness is a corpse.

## 1. What a core dump actually is

Before we capture or read one, we have to be exact about what a core dump *is*, because almost every confusion about post-mortem debugging is a confusion about the underlying machine. A running process is, at the level the kernel cares about, a chunk of virtual memory plus a set of CPU registers per thread. The memory holds your code, your global data, your heap, and one stack per thread. The registers hold the current instruction pointer (`RIP` on x86-64), the stack pointer, the frame pointer, and the general-purpose registers that contain whatever the CPU was computing at that instant. When the process is alive, those things change billions of times a second. A core dump is what you get when you **stop time** and write all of it to disk.

Concretely, when a process receives a signal whose default action is "terminate and dump core" — `SIGSEGV` (segmentation fault, signal 11), `SIGABRT` (abort, signal 6, what `assert` and `abort()` raise), `SIGBUS`, `SIGFPE`, `SIGILL`, `SIGQUIT` — the kernel does not just kill it. It walks the process's memory map and writes the contents of the writable, readable regions out to a file, in **ELF core format** on Linux (the same ELF container that holds executables, just with a different type). Inside that file are notes describing every thread's register set, the `siginfo` that says exactly which signal fired and at what faulting address, the mapped regions (which shared libraries were loaded and where), and the actual bytes of the stack and heap. The result is a snapshot. It is not a log of what happened; it is a photograph of the single instant the process died, accurate down to the byte.

![A vertical layered view of a core dump showing registers with the instruction pointer at the fault on top, then all thread stacks, then call frames with locals and arguments, then heap pages, then mapped library regions, and the faulting address at the bottom](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-2.png)

The figure above lists what is inside that photograph, and every layer matters in an investigation. The **registers** tell you the exact instruction that faulted and the values it was operating on. The **thread stacks** — one per thread — tell you what every thread was doing, which is how you find *which* thread crashed when nine of them are idle and one dereferenced a null pointer. The **call frames** within the crashed stack hold the local variables and arguments of every function on the path to the crash, which is the actual evidence: the pointer that was null, the index that was out of bounds, the length that was negative. The **heap pages** let you follow that null pointer's owner, or inspect the object it should have pointed to. The **mapped regions** tell the debugger where each shared library lived in memory, which is what lets it turn an address into `libc.so.6 + 0x9f4a3`. And the **faulting address** — the `si_addr` field of the signal info — tells you the precise memory address the program tried to touch when the hardware said no. A fault at `0x0` is a null dereference. A fault at `0x8` is a null pointer plus a small struct offset (you dereferenced `null->field`). A fault at a wild address like `0x6161616161616161` is `"aaaa..."` — a string overran a buffer and smashed a return address. The faulting address alone often tells you the bug *class* before you read a single frame.

Here is the mental shift that makes post-mortem debugging powerful, and it is worth stating plainly under this figure: **a core dump is a debugger session frozen in time.** Everything you could do with a live debugger attached at the moment of the crash — print a variable, walk up the stack, examine an arbitrary memory address, list which thread holds which lock — you can do against the core, *after the fact, on a different machine, days later.* The only thing you cannot do is step forward, because the process is dead and time has stopped. But you do not need to step forward. The crash already happened. You need to look *backward and around* at the state that produced it, and for that the core is a perfect, complete record. This is why the autopsy metaphor is exact and not just cute: the patient is dead, you cannot interview them, but the body holds every clue about how they died, and your job is to read the body.

### Why it crashed there but the bug is elsewhere

It helps to be precise about *when and how* the kernel writes the core, because it explains a few surprising behaviors. When the CPU executes an instruction that touches memory the process is not allowed to touch — a load through a null or wild pointer, a write to a read-only page — the hardware raises a **page fault**, and the kernel's fault handler examines it. If the address is part of a legitimate mapping that simply is not resident yet, the kernel pages it in and the instruction retries; this is the normal, invisible mechanism behind demand paging. But if the address belongs to no valid mapping, or violates the page's permissions, the kernel cannot satisfy it, and it delivers a `SIGSEGV` to the offending thread. The signal's `siginfo` carries the faulting address (`si_addr`) and a code saying whether it was a missing mapping (`SEGV_MAPERR`) or a permission violation (`SEGV_ACCERR`) — distinctions that survive into the core and are visible in `gdb`. If the process installed no handler for that signal, the default disposition runs: terminate and dump core. The kernel suspends every thread, walks the memory map, and serializes the readable regions to the core file *synchronously*, before the process is reaped. This synchronicity is why dumping a multi-gigabyte process takes real time and disk, and why a host under memory pressure can be pushed further toward the edge by the very act of capturing the evidence — a 4 GB-heap service writes a 4 GB core, and if the disk is nearly full the write itself can fail and you get a *truncated* core, the worst kind, where `gdb` opens it and warns "the file is truncated" and half your heap is missing. Sizing the core volume is part of making cores reliable, not an afterthought.

One subtlety worth internalizing: a `SIGABRT` core (from `abort()`, a failed `assert`, glibc's heap-corruption detector, or a C++ `std::terminate`) is just as useful as a `SIGSEGV` core, and often *more* useful, because the program chose to die at a clean, well-defined point rather than wandering into undefined behavior first. When glibc detects heap corruption — a double-free, a write past a malloc chunk's header — it prints `free(): invalid pointer` or `malloc(): corrupted top size` and calls `abort()`, which dumps core *at the moment corruption was detected*, with the corrupting call still near the top of the stack. That is a gift: the allocator caught the bug close to its source instead of letting it crash 10,000 lines later. So when you see `SIGABRT` with a glibc message, the autopsy is usually *easier* than a raw segfault, because the runtime did the localization for you.

One more piece of mechanism carries over from reading a live stack trace, and it is the single most important habit in post-mortem work: **the faulting instruction is where the program *detected* a problem, not where the mistake was made.** The CPU faulted in `strlen` because it was handed a null pointer. But `strlen` is library code that did exactly what it was told; the *bug* is up the stack, in your function that passed `null` to `strlen` because it never checked the return of a lookup that failed. The core gives you the crash site for free — that is the easy part, the kernel hands it to you. Finding the bug site is the reading skill, and it is the same skill as [reading a stack trace across languages](/blog/software-development/debugging/reading-a-stack-trace-across-languages): start at the crash, walk *up* the stack into your own code, and at each frame ask "what did this function pass down, and was it valid?" The difference with a core dump is that you do not just see the frame *names* — you can read the actual *argument values* in every frame, frozen at the moment of death. That is the superpower. A stack trace tells you the call chain; a core dump tells you the call chain *and every value in it*.

## 2. Making cores exist: the part everyone skips

The most common failure in post-mortem debugging is not misreading the core. It is that **there is no core to read.** The crash happened, the process died, and nothing was saved, because by default most systems do not write cores — or write them somewhere you never look, or truncate them to zero bytes because a limit was set to zero a decade ago in some init script nobody remembers. Before you can do any autopsy, you have to make the corpse exist. This is plumbing, it is unglamorous, and it is the difference between root-causing the crash and shrugging.

On Linux the core machinery has two knobs. The first is a **resource limit**: every process has a maximum core-file size, and if it is `0`, no core is written no matter what. You raise it with `ulimit`:

```bash
# Check the current limit (very often this prints 0)
ulimit -c
# Allow cores of unlimited size for this shell and its children
ulimit -c unlimited
# Make it stick for a service: in the systemd unit
#   [Service]
#   LimitCORE=infinity
```

The second knob is **where the core goes**, controlled by `/proc/sys/kernel/core_pattern`. This is a template string the kernel expands when it writes a core. If it is the literal string `core`, you get a file named `core` in the process's current working directory — which for a daemon is often `/` and not writable, so the core silently fails. Modern distros pipe it to a handler instead. You read and set it like this:

```bash
# See where cores go on this host
cat /proc/sys/kernel/core_pattern
# A sane explicit pattern: a named file with pid, exe, and signal
echo '/var/cores/core.%e.%p.%s.%t' | sudo tee /proc/sys/kernel/core_pattern
mkdir -p /var/cores && chmod 1777 /var/cores
# %e=exe name  %p=pid  %s=signal number  %t=unix time  %h=hostname
```

On a systemd host the pattern is usually `|/usr/lib/systemd/systemd-coredump`, which pipes the core into the journal-managed core store. That is good, because it handles rotation and compression for you, and it gives you a clean front-end called `coredumpctl`:

```bash
# List recent crashes systemd captured
coredumpctl list
# Show details (signal, exe, build-id, the command line) for the latest
coredumpctl info
# Open the most recent core for a given program directly in gdb
coredumpctl debug myservice
# Or extract the core file to disk to copy elsewhere
coredumpctl dump myservice --output=/tmp/myservice.core
```

`coredumpctl debug` is the single most ergonomic way to start an autopsy on a modern Linux box: it finds the core, finds the executable, loads both into `gdb`, and drops you at the prompt. When it works, it removes every excuse. When it does not — because the service ran in a container, or the binary was deleted on redeploy — you fall back to the manual path, which is the next two sections.

### Snapshotting a live process without killing it

Sometimes the process has not crashed — it is *hung*, or *leaking*, or behaving strangely, and you want a snapshot to analyze without taking the process down. That is what `gcore` is for. It attaches to a running process, writes out a core image, and detaches, leaving the process running:

```bash
# Find the pid and snapshot it; the process keeps running
gcore -o /tmp/hung 4127      # writes /tmp/hung.4127
# Now open the snapshot offline while prod keeps serving
gdb /path/to/binary /tmp/hung.4127
```

This is invaluable for a deadlock: you cannot reproduce a deadlock on demand, but when one host is wedged you can `gcore` it, get a snapshot of every thread's stack and which lock each is blocked on, and analyze the deadlock at your leisure while you fail the host over. It is the post-mortem technique applied to a process that is not yet dead — a *living autopsy*. The same idea underlies the sibling discussion of [deadlocks, livelocks, and starvation](/blog/software-development/debugging/deadlocks-livelocks-and-starvation), where the thread dump is the central artifact.

### The per-runtime switches

Native code on Linux gets you a core from the kernel for free once the limits are set. Managed runtimes are different: they often *catch* the fatal condition themselves, print a stack trace, and exit cleanly — which is friendlier day to day but means no core, and a managed stack trace alone cannot show you native heap state or an OOM's retained objects. Each runtime has its own switch to force a usable dump.

![A matrix mapping native C and C++, Go, Java, .NET, and Python each to the one switch that captures a dump and the tool you read that dump with](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-4.png)

The figure above is the cheat sheet; here is what each row means in practice. In **Go**, a panic prints a goroutine trace and exits, but it does not write a core unless you set `GOTRACEBACK=crash` in the environment, which turns the final panic into a `SIGABRT` that the kernel dumps:

```bash
# Make a Go panic produce a real core file the kernel can dump
GOTRACEBACK=crash ./myserver
# then analyze with Delve's core mode (next sections)
dlv core ./myserver /var/cores/core.myserver.4127.6.1718900000
```

In the **JVM**, the crash you most often post-mortem is an `OutOfMemoryError`, and the switch that saves you is one that writes a heap dump *at the moment the heap is exhausted*, before the JVM unwinds and loses the state:

```bash
# Capture a heap dump automatically the instant the heap is exhausted
java -XX:+HeapDumpOnOutOfMemoryError \
     -XX:HeapDumpPath=/var/cores/heap.hprof \
     -jar app.jar
# Or grab one on demand from a live, struggling JVM:
jmap -dump:live,format=b,file=/tmp/live.hprof <pid>
```

In **.NET**, `dotnet-dump` collects a full process dump on demand or on crash, and analyzes it with SOS commands:

```bash
# Collect a dump from a running .NET process
dotnet-dump collect -p <pid> -o /var/cores/app.dmp
# Or configure crash dumps via env vars:
#   DOTNET_DbgEnableMiniDump=1
#   DOTNET_DbgMiniDumpType=4    (full)  DbgMiniDumpName=/var/cores/app.dmp
```

In **Python**, a pure-Python exception prints a traceback and is not a "crash" in the core sense; the crashes that *do* produce a core are segfaults inside C extensions (NumPy, a database driver, your own `ctypes` call). `faulthandler` prints the Python half of such a crash, and you still want a real core for the C half:

```python
import faulthandler
faulthandler.enable()      # prints a Python traceback on SIGSEGV/SIGFPE/SIGABRT
# Run under ulimit -c unlimited so you also get a core for the native frames.
# Read the core with gdb plus the python-gdb extension:
#   gdb python3 core.python3.4127.11
#   (gdb) py-bt        # Python-level backtrace
#   (gdb) bt           # native C-level backtrace
```

That last note — `py-bt` for the Python frames and `bt` for the C frames in the *same* core — is the crux of debugging a mixed-language crash, and it is the same "two unwinders" reality from the stack-trace post: the managed runtime tracks Python frames precisely, the native unwinder needs symbols for the C frames, and a full autopsy needs both. We will see the JVM equivalent (a heap dump for managed state, a core for native state) in the OOM worked example.

## 3. Symbolization: why a stripped binary says `?? ()`

You have a core. You open it in `gdb`. You type `bt`. And you get this:

```bash
(gdb) bt
#0  0x000055f3a1 in ?? ()
#1  0x000055f410 in ?? ()
#2  0x000055f4d2 in ?? ()
#3  0x000055f5a9 in ?? ()
#4  0x00007f9c12 in ?? () from /lib/x86_64-linux-gnu/libc.so.6
# ... nine frames, every one of them ?? () — no function, no file, no line.
```

Every frame is a raw hexadecimal address followed by `?? ()`, with no function name, no file, and no line number. Nine frames of `?? ()`. A dead end. This is the second most common way post-mortem debugging fails after "no core at all," and understanding *why* it happens tells you exactly how to fix it.

A compiled binary is machine code: a sequence of instructions at numeric addresses. The mapping from an address like `0x55f3a1` back to "this is the function `renew_token`, defined at `session.c` line 212" lives in the **debug symbols** — the DWARF debug information (`.debug_info`, `.debug_line`, and friends) that the compiler emits when you build with `-g`. That information is *large*, often several times the size of the code itself, so for production you usually **strip** it: `strip` removes the symbol table and debug sections to shrink the binary and to avoid shipping source-revealing metadata. A stripped binary runs identically — the machine code is untouched — but it has thrown away the dictionary that translates addresses back into names. When `gdb` opens a core produced by a stripped binary, it has the addresses (they are in the core) but no dictionary, so every frame is `?? ()`. The crash is real and fully captured; you just cannot *read* it.

![A two-column before and after where the left shows a stripped binary giving address-only frames that dead-end, and the right shows matching the build-id to a separate debug file that names each frame down to a file and line](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-3.png)

The fix, shown on the right of the figure above, is to keep the debug information — just not *inside* the shipped binary. The standard pattern is to split the symbols into a separate file at build time and keep it in a **symbol server** (or just an artifact store) keyed by build:

```bash
# 1. Build with debug info (-g), then split symbols out of the binary
objcopy --only-keep-debug myapp myapp.debug
# 2. Strip the shipped binary, but leave a link to the debug file
strip --strip-debug --strip-unneeded myapp
objcopy --add-gnu-debuglink=myapp.debug myapp
# 3. Archive myapp.debug keyed by its build-id; ship the stripped myapp.
# 4. At autopsy time, point gdb at the debug file:
gdb myapp /var/cores/core.myapp.4127.11
#   (gdb) symbol-file myapp.debug    # or set debug-file-directory
```

The crucial word is **build-id**. Every compiled binary carries a unique `build-id` — a hash of its contents — and the core dump records the build-ids of the executable and every shared library that was loaded. The debug file carries the *same* build-id as the binary it came from. This is what guarantees you symbolize against the *exact* build that crashed, not a similar one. If you symbolize a core from build A using the debug symbols of build B, every address resolves to the wrong line, and you will confidently root-cause a bug that does not exist — the worst outcome in debugging, because you will "fix" working code and ship a new bug. You check the match like this:

```bash
# The build-id baked into the binary
file myapp                       # ... BuildID[sha1]=4a2f...e91
readelf -n myapp | grep -A1 'Build ID'
# The build-ids the core expects (gdb prints mismatches loudly):
gdb myapp core   ->  "warning: the debug information found ... does not match"
```

When you only have raw addresses and no `gdb` session — for example a crash reporter uploaded just the faulting addresses — you symbolize a single address with `addr2line`:

```bash
# Turn a raw address from a crash report into file:line
addr2line -e myapp.debug -f -C 0x55f3a1
#   renew_token
#   /src/session.c:212
# -f prints the function, -C demangles C++ names, -e picks the binary/debug file
```

This is exactly the mechanism that field crash reporters rely on. **Breakpad**, **Crashpad**, and **Sentry** ship a tiny in-process handler that, on a crash, writes a compact **minidump** (a Windows-originated format that is far smaller than a full core — it captures the stacks and a slice of memory, not the whole heap) and uploads it. The server then symbolizes the minidump *server-side* against a symbol store keyed by build-id, exactly as above, so the developer sees `renew_token at session.c:212` even though the user's machine only ever had a stripped binary and never saw a line of your source. We will come back to minidumps and field collection in the triage section, because symbolizing one crash and bucketing ten thousand are two halves of the same problem.

## 4. Reading the core in gdb: the autopsy

Now the core exists and the symbols match. This is the autopsy itself, and it is a small, repeatable sequence of commands. I will walk a real SIGSEGV from open to root cause, and the sequence here is the one you should burn into muscle memory because it works for almost every native crash.

![A left-to-right timeline of a gdb session from loading the core, applying backtrace to every thread to find the crashed one, running backtrace full, moving to your own frame, reading info locals to find a null pointer, and arriving at the root cause](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-5.png)

The figure above is the order of operations. Step one is to open the core with its binary, and the very first thing `gdb` prints is gold — it tells you the signal and often the faulting function:

```bash
$ gdb ./myapp /var/cores/core.myapp.4127.11
Reading symbols from ./myapp...
[New LWP 4127]
[New LWP 4131]
Core was generated by `./myapp --serve'.
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x000055f3a1 in strlen () at ../sysdeps/x86_64/strlen.S:120
(gdb)
```

`SIGSEGV` in `strlen` — already we know the bug class: something handed `strlen` a bad pointer. Step two, when the program is multithreaded, is to find *which* thread actually crashed, because `gdb` may drop you on the wrong one. `thread apply all bt` prints a backtrace for every thread:

```bash
(gdb) info threads
  Id   Target Id         Frame
* 1    LWP 4127          0x000055f3a1 in strlen ()
  2    LWP 4131          0x00007f... in __futex_abstimed_wait () # idle, blocked
(gdb) thread apply all bt
```

Thread 2 is parked in a futex wait — idle, waiting for work, innocent. Thread 1 is the one in `strlen`. We switch to it (it is already current here) and run the full backtrace, `bt full`, which prints every frame *with its local variables*:

```bash
(gdb) bt full
#0  0x000055f3a1 in strlen () at strlen.S:120
No locals.
#1  0x000055f410 in log_user (s=0x0) at log.c:44
        prefix = 0x55f900 "user="
#2  0x000055f4d2 in handle_request (req=0x7ffd1a) at handler.c:307
        sess = 0x0
        uid = 91182
#3  0x000055f5a9 in worker_loop () at worker.c:88
        ...
```

Read it from the bottom up into your code, exactly as you would a stack trace. Frame 0 (`strlen`) is libc — innocent. Frame 1 (`log_user`) is library-ish glue — it received `s=0x0` and dutifully passed it to `strlen`. Frame 2, `handle_request` at `handler.c:307`, is *your code*, and look at its locals: `sess = 0x0`. The session pointer is null. The bug is here: `handle_request` looked up a session, got null (the session expired and was freed), did not check, and called `log_user(sess->name)` — which is `log_user(0x0->name)`, a read at offset `0x0`, the null dereference. We did not reproduce anything. We read it off the corpse. Step three of confirming this is to move to that frame and interrogate it directly:

```bash
(gdb) frame 2
#2  handle_request (req=0x7ffd1a) at handler.c:307
307         log_user(sess->name);
(gdb) info args
req = 0x7ffd1a
(gdb) info locals
sess = 0x0
uid = 91182
(gdb) print sess
$1 = (struct session *) 0x0
(gdb) info registers rdi
rdi  0x0    0          # the argument register held null at the call
```

`info args` shows what `handle_request` was called with; `info locals` shows what it had computed; `print sess` confirms the null; `info registers` shows the raw register state at the fault. That is the complete chain of evidence — symptom (SIGSEGV), site (`strlen`), and cause (`sess` was null at `handler.c:307`) — assembled from a dead process without a single reproduction. Below is a compact reference of the gdb commands that do the actual work in an autopsy:

| gdb command | What it answers | When to reach for it |
| --- | --- | --- |
| `bt` / `bt full` | The call chain; `full` adds locals per frame | First, always; `full` to see the bad value |
| `thread apply all bt` | What every thread was doing | Multithreaded crash, deadlock, or wrong-thread drop |
| `frame N` | Move to frame N to inspect it | After `bt` identifies your frame |
| `info args` / `info locals` | Arguments and locals of the current frame | To read the actual bad value |
| `info registers` | Raw register state (RIP, RDI, RSP...) | Confirm the faulting argument and instruction |
| `print expr` / `p *ptr` | Evaluate an expression against the corpse | Dereference structs, follow pointers, read fields |
| `x/NFU addr` | Examine N units of memory at an address | Inspect the faulting address or a buffer |
| `info sharedlibrary` | Loaded libraries and their build-ids | Diagnose symbol mismatches |

#### Worked example: a SIGSEGV in prod you never saw

Here is the full investigation, with the kind of detail you would actually hit. A payments worker crashed once at 02:47 on a Saturday and never again. The dashboard said signal 11, the process restarted, and there was nothing in the logs but the request id `req-7f3a91`. We had `LimitCORE=infinity` and a `core_pattern` of `/var/cores/core.%e.%p.%s.%t`, so when on-call checked Monday there was indeed a file: `core.payments.20194.11.1718852820`, 612 MB. The binary was stripped (production), but we had archived `payments.debug` keyed by build-id in our artifact store.

First, confirm the symbol match so we do not chase a phantom:

```bash
$ readelf -n /opt/payments/bin/payments | grep -A1 'Build ID'
    Build ID: 4a2f8c1d...e91
$ readelf -n payments.debug | grep -A1 'Build ID'
    Build ID: 4a2f8c1d...e91          # identical — safe to symbolize
```

Then open and walk it:

```bash
$ gdb /opt/payments/bin/payments core.payments.20194.11.1718852820
...
Program terminated with signal SIGSEGV, Segmentation fault.
#0  0x000055a1 in account_currency (acct=0x18) at account.c:55
(gdb) bt full
#0  account_currency (acct=0x18) at account.c:55
#1  settle (txn=0x7f12, acct=0x18) at settle.c:120
        amount = 4200          # cents
#2  process_refund (txn=0x7f12) at refund.c:73
        acct = 0x18            # <-- look at this
        orig = 0x7f12
(gdb) frame 2
(gdb) print acct
$1 = (struct account *) 0x18
```

The faulting address was `0x18` — not `0x0`. That is the tell: `0x18` is `0x0 + 0x18`, a null pointer plus the byte offset of the `currency` field inside `struct account`. So `acct` was null, and `account_currency` read `acct->currency` at offset 24. Why was `acct` null? We followed it up: `process_refund` got `acct = 0x18` too, so the null came from *its* caller. We looked at `refund.c:73`:

```c
struct account *acct = lookup_account(txn->account_id);
settle(txn, acct);   /* line 73-ish: acct never checked */
```

`lookup_account` returns null when the account was closed, and a refund for a *closed* account — which only happens for transactions older than the 90-day retention window, which only happens on a weekend batch — slipped through with a null `acct` that nobody checked. We confirmed by reading `txn` out of the core:

```bash
(gdb) print *txn
$2 = {account_id = 88120033, amount = 4200, created = 1710000000, ...}
(gdb) print orig->created
$3 = 1710000000      # ~98 days before the crash — past retention
```

Root cause: a missing null check on `lookup_account` in the refund path, triggered only by a transaction past the retention window, which is why it fired once on a weekend and never in any test. The fix was four lines (check for null, skip and alert on a closed account). The proof it was right: we wrote a unit test that calls `process_refund` with an account id that `lookup_account` reports closed, watched it segfault on the old code, applied the fix, and watched it pass — and then we re-derived the same `0x18` faulting address in a local core to be certain we had explained *this* crash and not a similar one. Total time from "we have a core" to "root cause confirmed": about twenty-five minutes, on a bug that had defeated every attempt to reproduce it by replaying traffic.

## 5. Examining memory and the faulting address

The autopsy in the last section read pointers and structs. Sometimes the pointer is not cleanly null — it is *garbage*, a wild value, and the question becomes "what wrote garbage here?" That is where memory examination earns its place, and it is the technique that separates a use-after-free from a plain null dereference.

![A three-by-three grid of memory-examination steps starting from printing a zero pointer and the faulting address, dumping the bytes before it to find a freed allocator poison pattern, then walking up to find the double-close on the error path that caused a use after free](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-8.png)

The grid above is the shape of a use-after-free autopsy. The key tool is `x`, the examine-memory command, which dumps raw bytes at any address in the core. Suppose the faulting pointer was not `0x0` but `0x6261645f70747200` — that is ASCII `"bad_ptr\0"` read as a little-endian integer, which means something wrote a *string* over a pointer slot: a buffer overran into an adjacent pointer. You confirm by examining the bytes:

```bash
(gdb) print sess
$1 = (struct session *) 0x6261645f70747200
(gdb) x/8c &sess          # examine 8 chars at the slot's address
0x7ffd...:  98 'b'  97 'a'  100 'd'  95 '_'  112 'p'  116 't'  114 'r' 0 '\0'
```

The pointer slot literally contains the characters `bad_ptr`. Some earlier write ran past the end of a buffer and clobbered `sess`. Now you know it is a buffer overflow, not a null bug, and the [use-after-free and memory corruption](/blog/software-development/debugging/use-after-free-and-memory-corruption) sibling post picks up that thread. A different and very common signature is the **allocator poison pattern**. Many allocators (and debug builds) fill freed memory with a recognizable pattern so that a use-after-free *looks* obviously wrong: glibc's `MALLOC_PERTURB_`, the classic `0xdeadbeef`, MSVC's `0xdddddddd` ("dead"), `0xfeeefeee` for freed heap. If you examine a struct in the core and every field is `0xdeadbeef`, the object was *freed* and you are looking at a use-after-free:

```bash
(gdb) print *sess
$2 = {refcount = 0xdeadbeef, name = 0xdeadbeef, token = 0xdeadbeef, ...}
# Every field is the freed-poison pattern: this object was already free()d.
```

The presence of the poison is the proof. The object was alive, it was freed (the allocator scribbled the pattern over it), and then this code path used it anyway — a use-after-free. From there you walk *up* the stack to find who freed it: which is usually a double-free or a free-on-error-path that also runs on the success path. In the grid, that is the "double close on the error path" node — a `close()`/`free()` in an error handler that also reached the normal path, freeing the session twice and leaving a dangling pointer that the next request reused. The deeper mechanism is worth stating because it explains why these crash "10,000 lines later": when you `free()` a block, the allocator does not erase it or return it to the OS; it puts it on a free list to hand out to the *next* allocation. So a freed object keeps its old bytes until something else allocates and overwrites it — which is why a use-after-free can read stale-but-plausible data for a while and then crash much later when a *different* allocation reuses and rewrites that block. The core freezes whichever state you happened to catch. If you are quick and lucky, the poison is still there and the diagnosis is instant; if not, you see the *new* tenant's data in the old object and have to reason about the reuse. This non-determinism is exactly why use-after-free bugs are heisenbugs that vanish under a debugger — a topic the [heisenbugs that vanish when you look](/blog/software-development/debugging/heisenbugs-that-vanish-when-you-look) sibling covers.

A few more memory-examination formats you will reach for constantly:

```bash
(gdb) x/16xb 0x7ffd1000      # 16 bytes in hex, byte by byte
(gdb) x/8xg $rsp             # 8 giant (8-byte) words from the stack pointer
(gdb) x/i $rip               # disassemble the faulting instruction itself
(gdb) x/s 0x55f900           # read a C string at an address
(gdb) p/x $rdi               # the first integer-argument register in hex
(gdb) p sizeof(struct session)   # confirm the offset math on a field
```

The `x/i` examine-as-instruction form pointed at the instruction-pointer register is underrated: it disassembles the exact instruction that faulted, so you can see, for instance, `mov (%rax),%rdx` with `rax = 0x0` and know it was a load through a null base register — the hardware-level confirmation of a null dereference, below even the C source line.

#### Worked example: a Python segfault inside a C extension

A data pipeline written in Python crashed with nothing but `Segmentation fault (core dumped)` and exit code 139 (which is `128 + 11`, the shell's encoding of "killed by signal 11"). A pure-Python program does not segfault — a Python bug raises an exception with a clean traceback — so a segfault means the crash was in *native* code: a C extension, a wheel with compiled internals, or a `ctypes` call. The traceback was empty because the C code died before Python could unwind. This is the canonical "two unwinders" crash, and it needs both halves.

We had set `faulthandler.enable()` at startup, so the process did print one thing before dying — a Python-level traceback showing where in *Python* the call into native code happened:

```python
Fatal Python error: Segmentation fault

Current thread 0x00007f3a (most recent call first):
  File "transform.py", line 88 in _decode_frame   # called into the C extension here
  File "transform.py", line 51 in process
  File "pipeline.py", line 203 in run
```

That tells us *which Python call* entered the native code (`_decode_frame` at `transform.py:88`), but not what went wrong *inside* the extension. For that we need the core, opened in `gdb` against the `python3` binary, using the CPython gdb extension that adds Python-aware commands:

```bash
$ gdb $(which python3) core.python3.30412.11
(gdb) bt                  # the NATIVE C backtrace
#0  0x00007f12 in decode_block (buf=0x0, n=4096) at codec.c:140
#1  0x00007f20 in _decode_frame_impl (self=..., args=...) at extension.c:88
#2  0x00005561 in cfunction_call () at Objects/methodobject.c:...
...
(gdb) py-bt               # the PYTHON backtrace from the SAME core
Traceback (most recent call first):
  File "transform.py", line 88, in _decode_frame
  File "transform.py", line 51, in process
(gdb) frame 0
(gdb) print buf
$1 = (unsigned char *) 0x0
```

The native `bt` shows `decode_block` faulted because `buf=0x0` — the extension was handed a null buffer. Walk up: `_decode_frame_impl` got a Python object and extracted a buffer pointer from it without checking that the underlying object was non-empty. The Python side passed an empty frame (a zero-length `bytes`) that the extension's C code dereferenced without a guard. Root cause: a missing length check at the C boundary, triggered only by an empty input frame, which only appeared in one malformed upstream file. The proof was a one-line reproduction — `transform.process(b"")` — that segfaulted on the old extension and raised a clean `ValueError` after we added the guard. The lesson that generalizes: for any crash that straddles managed and native code, you need `py-bt` (or the JVM's mixed-frame handling) for the managed half and `bt` for the native half, *both against the same core*, because no single unwinder spans the boundary. The faulthandler output narrows you to the call site; the core's native frames tell you what happened past it.

## 6. The autopsy in other runtimes: dlv, MAT, dotnet-dump

`gdb` is the native autopsy tool, but the corpse looks different in a managed runtime, and so do the tools. The *principle* is identical — freeze the dead state, find the crashed thread, walk to your frame, read the bad value — but the commands change.

**Go** uses Delve in core mode. With `GOTRACEBACK=crash` set, a panic produces a real core, and `dlv core` reads it with Go-aware commands that understand goroutines, channels, and Go types:

```bash
$ dlv core ./myserver /var/cores/core.myserver.20871.6.1718900000
(dlv) goroutines          # every goroutine, not OS threads
(dlv) goroutine 18        # switch to the one that panicked
(dlv) stack               # its Go backtrace
(dlv) frame 2
(dlv) locals              # Go locals, with Go types
(dlv) print sess          # *Session = nil
```

The win over `gdb` here is that Delve speaks Go: it shows you goroutines (Go's lightweight threads) rather than OS threads, it formats Go maps and slices and channels properly, and it can show you a blocked goroutine's channel — which is how you post-mortem a Go deadlock from a core. The grouping-by-goroutine is the same idea as the per-goroutine stack-trace format from the stack-trace post, now navigable in a dead process.

**Java's OOM** is the headline post-mortem case, and the tool is a **heap dump** plus Eclipse MAT (Memory Analyzer Tool), not a core. The heap dump (`.hprof`) is a snapshot of every live object on the Java heap with the references between them. MAT's superpower is two analyses built on the **dominator tree**.

![A two-column before and after where the left shows a raw heap dump of millions of objects with a flat histogram you cannot blame, and the right shows MAT's dominator tree and leak suspects pointing at one HashMap retaining 1.8 gigabytes through a static cache](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-7.png)

The figure above contrasts the raw dump with what MAT extracts. The naive view of a heap dump is a **histogram**: how many of each class, and how much *shallow* memory each occupies. That view is almost useless for a leak, because the top of the histogram is always `byte[]`, `char[]`, `String` — the leaf objects, not the thing *holding* them. The breakthrough concept is **retained size**: the amount of memory that would be freed if a given object were garbage-collected, i.e. the object *plus everything only it keeps alive*. The **dominator tree** is the structure that computes this — object X *dominates* object Y if every path from the GC roots to Y goes through X, so X is the thing truly responsible for Y's memory. Sort the dominator tree by retained size and the leak culprit floats to the top: not the millions of `String`s, but the *one* `HashMap` that retains all of them. MAT's "Leak Suspects" report automates exactly this and writes you a paragraph naming the dominator.

The flow in MAT is short and always the same. Open the `.hprof`, run **Leak Suspects**, and read the one suspect it names — something like "one instance of `java.util.HashMap` retains 1.8 GB." Open that object's **dominator tree** entry, then right-click → **Path to GC Roots → exclude weak/soft references** to see *why* the garbage collector cannot reclaim it (it turns out to be held by a `static` field — an application-lifetime cache that never evicts). Finally, right-click → **List objects → with incoming references** to see what is actually *in* the map, which is where you discover that every key is unique so nothing was ever a cache hit. That is the leak found, from a corpse, with no reproduction. We work that example fully below.

**.NET** uses `dotnet-dump analyze`, whose command vocabulary comes from SOS (the long-standing CLR debugging extension):

```bash
$ dotnet-dump analyze /var/cores/app.dmp
> clrthreads               # managed threads; find the one with an exception
> clrstack                 # managed call stack of the current thread
> clrstack -a              # with arguments and locals
> printexception           # the exception that crashed it, with its stack
> dumpheap -stat           # heap histogram by type (the .NET equivalent)
> gcroot <address>         # why is this object still alive? (path to roots)
> dumpobj <address>        # dump a managed object's fields
```

`gcroot` is the .NET analogue of MAT's "path to GC roots": you hand it the address of an object you suspect is leaking and it tells you the chain of references keeping it alive, which is how you find a .NET memory leak from a dump. Across all three runtimes the question is the same — *what is holding this memory, and why can't the GC reclaim it?* — and the answer is always a reachability path from a root to the leaked object. A core dump (or heap dump) is the only artifact that contains that whole graph.

#### Worked example: an OOM crash, root-caused from a heap dump

A reporting service kept dying with `java.lang.OutOfMemoryError: Java heap space` after about six hours of uptime, only in production, never in staging. It was launched with `-Xmx4g -XX:+HeapDumpOnOutOfMemoryError -XX:HeapDumpPath=/var/cores/heap.hprof`, so the OOM left behind `heap.hprof`, 4.1 GB on disk. We copied it off the host and opened it in MAT. Here is the chain of reasoning and the numbers.

The histogram (the wrong view) showed `char[]` at the top with 2.6 GB shallow — uninformative, because every leak ends in character arrays. We ignored it and ran **Leak Suspects**, which reported one suspect immediately: a single instance of `java.util.concurrent.ConcurrentHashMap` with a **retained size of 1.8 GB** — 44% of the 4 GB heap held by one object. We opened its dominator tree entry and ran **Path to GC Roots, excluding soft and weak references**, which is the question "why can't this be collected?" The path ended at a `static` field: `ReportCache.INSTANCE.cache`. A process-lifetime static cache, never evicting. Then **List objects with incoming references** on the map showed 1.9 million entries keyed by a `ReportRequest` whose `equals`/`hashCode` included a timestamp — so every request was a *new* key, nothing was ever a cache *hit*, and the "cache" grew without bound for the life of the process.

That explained every symptom precisely. It only OOM'd in production because only production had the request volume to add 1.9 million entries before a deploy restarted the JVM and reset the heap; staging never ran long enough or hot enough. It took ~6 hours because that is how long ~1.9 million entries at the production rate took to fill 1.8 GB. The fix was to bound the cache (an LRU with a 50,000-entry cap and a TTL) so retained size stayed flat. The proof: after the fix we watched the old-generation occupancy on the prod JVM over 24 hours — before, it climbed monotonically toward 4 GB and crashed around hour six; after, it sawtoothed between 1.1 GB and 1.6 GB indefinitely, with GC reclaiming the evicted entries. The leak went from "+~300 MB/hour until death" to flat. We found a leak that had killed the service for weeks, from a single corpse, in under an hour, without ever reproducing the OOM ourselves. This is the same retained-size reasoning the [hunting memory leaks and bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) sibling applies to live processes; the difference is that here the process was already dead and the dump was all we had.

## 7. Cores from containers and from the field

Two situations break the simple "find the core in `/var/cores`" story, and both are now the common case: the crash happened *inside a container*, or it happened *on a user's machine in the field*. Each needs a different collection strategy, and getting this wrong means the crash happens and no core survives.

The container surprise catches everyone once: **`/proc/sys/kernel/core_pattern` is host-wide, not per-container.** A container shares the host kernel, so when a process inside a pod crashes, the *host's* `core_pattern` decides where the core goes — not anything in your container's filesystem or your Dockerfile. If the host pattern pipes to `systemd-coredump`, the core lands in the *host's* core store, owned by the node, not visible inside the dead pod (which is gone anyway). If the host pattern is a relative `core`, the kernel tries to write into the container's working directory, which is usually a read-only or ephemeral layer that vanishes when the pod dies. Either way, the naive expectation — "the core will be in my container" — fails. The robust patterns:

| Strategy | How it works | Trade-off |
| --- | --- | --- |
| Host `core_pattern` to a shared path | Set the node's pattern to write to a hostPath the pod mounts | Needs node access; per-node config drift |
| `systemd-coredump` on the node + sidecar pull | Cores land in the node's coredumpctl store; a DaemonSet extracts and uploads them | Centralized, but you debug off-node |
| In-process crash handler (Breakpad/Crashpad) | The app writes its own minidump to a mounted volume on crash | Works without node access; minidump not full core |
| `gcore` from a privileged debug container | Snapshot a live/hung pod's process via `kubectl debug` | Manual; great for hangs, not for already-dead pods |

The cleanest production setup is usually the second: configure the node's `core_pattern` once (via the node image or a privileged DaemonSet that writes `/proc/sys/kernel/core_pattern` at boot) to a known location, and run a small DaemonSet that watches that location, compresses each core, attaches the pod and build metadata, and uploads it to object storage keyed by build-id. Then the debugger downloads the core and the matching debug symbols and does the autopsy off-node. Resource limits matter too: the pod's process needs `LimitCORE=infinity` (or the container runtime's equivalent `ulimit` setting), and the node needs disk headroom for cores, which for a 4 GB-heap service is a 4 GB file per crash. This composes with the [observability for debugging prod](/blog/software-development/debugging/observability-for-debugging-prod) sibling: the trace and logs tell you *that* and *roughly where* a pod crashed; the captured core tells you *exactly why*.

The **field** case — your software crashing on machines you do not own — is what minidumps and crash reporters were invented for. You cannot collect a 600 MB core from a million laptops, and you would not want the user's heap (privacy). So **Breakpad** and **Crashpad** (the handlers behind Chrome, Firefox, and countless desktop apps) install a signal/exception handler that, on a crash, writes a compact **minidump**: the crashed thread's stack, the other threads' stacks, the register state, the loaded-module list with build-ids, and a small slice of memory — typically tens to hundreds of kilobytes, not hundreds of megabytes. The handler uploads the minidump, and your **symbol server** symbolizes it against the build-id, so you get a real backtrace into your source without the user ever having symbols or source. **Sentry** wraps this same Breakpad/Crashpad/minidump pipeline with a UI and the triage features we cover next. The trade-off is fidelity: a minidump has the stacks and registers (enough for most crashes) but not the whole heap, so it cannot do the MAT-style retained-size leak analysis a full core can — for a leak you still want a full heap dump from one reproducing instance. Stacks-and-registers for the field; full heap for the leak. Choose by the bug.

## 8. Triaging crashes at scale: 10,000 reports, 5 bugs

A single core is an autopsy. A million crash reports a week is a *different* problem, and the skill flips from "read one corpse deeply" to "group ten thousand corpses into the handful of distinct bugs that produced them." Without grouping, a crash reporter is a firehose of noise: the same null dereference reported by 6,000 users looks like 6,000 problems, and a rare one-off looks the same size as a pandemic. The grouping technique is **crash signature bucketing**, and it is what makes field crash reporting usable.

![A tree showing ten thousand raw crash reports bucketing by stack signature into three distinct signatures, with the largest bucket of 6200 reports flagged to fix first and a long tail of rare crashes](/imgs/blogs/reading-a-core-dump-post-mortem-analysis-6.png)

The figure above is the core idea: hash each crash by a **normalized version of its top frames** and crashes that share a root cause collapse into one bucket. The normalization is the craft. You take the top N frames (often 3–5), strip the parts that vary between machines and builds — exact addresses, inlined library frames, anonymous-lambda numbers, line offsets that shift between versions — and keep the stable signal: the sequence of function names. Hash that into a **stack signature**. Two crashes with the same signature are, with high probability, the same bug. Now 6,000 identical null-in-`renew_token` crashes become *one bucket with a count of 6,000*, and your screen shows five buckets, not ten thousand rows. The win is enormous and it is the whole reason crash dashboards work:

| Without bucketing | With signature bucketing |
| --- | --- |
| 10,000 individual reports | ~5 distinct signatures |
| Every crash looks equally urgent | Ranked by frequency × severity |
| Re-investigate the same bug 6,000× | Investigate once per bucket |
| A regression is invisible in the flood | A *new* signature spikes and is obvious |
| No sense of impact | "6,200 users / 44% of crashes" per bucket |

This is the mindset behind Windows' `!analyze` (the WinDbg command that auto-classifies a dump, guesses the faulting module, and produces a bucket id) and behind every modern crash service. Once crashes are bucketed, you prioritize by **frequency × severity**, and the two are different axes. A crash that hits 6,200 users but only on a rarely-used export button is high-frequency, low-severity. A crash that hits 50 users but corrupts their saved data is low-frequency, catastrophic-severity. You want both numbers per bucket, and you fix the top-right of that quadrant first. A subtle, vital benefit: bucketing makes **regressions visible**. When a *new* signature appears after a deploy and climbs fast, it stands out against the stable background of known buckets — which turns crash triage into a release-quality gate. You ship a build, you watch for new signatures spiking, and a single new bucket at 1% and rising tells you to roll back before it reaches everyone. That is the bisection instinct applied to crashes: a new bucket correlated with one deploy is your first hypothesis, and [binary search your bug with bisection](/blog/software-development/debugging/binary-search-your-bug-with-bisection) is how you nail the exact commit.

#### Worked example: bucketing a release's crash spike

A desktop app shipped version 4.2 on a Tuesday. By Wednesday the crash service showed the raw count up 3× — about 11,400 new crash reports in 24 hours, which as an undifferentiated list was unreadable. Bucketed by stack signature, those 11,400 reports collapsed into 7 signatures, and the distribution was lopsided: one signature accounted for 6,200 of them (54%), a brand-new signature that did *not* exist in 4.1. Its top frames, after normalization, were `RenderCache::evict → Texture::release → GpuBuffer::~GpuBuffer`, and the minidumps all showed a faulting read at a poison address — a use-after-free in the texture eviction path. The second-largest bucket, 2,900 reports, was an *old* signature that had existed at the same rate in 4.1, so it was pre-existing background noise, not part of the regression. The remaining ~2,300 spread across five rare signatures, the long tail.

The triage decision wrote itself: the 6,200-report new signature was the regression, severity high (a hard crash losing unsaved work), and it correlated perfectly with the 4.2 deploy. We pulled one representative minidump, symbolized it against the 4.2 build-id, and got `Texture::release at texture.cpp:88` reading a freed `GpuBuffer`. `git log` between 4.1 and 4.2 on `texture.cpp` showed exactly one change: a refactor that moved the buffer free earlier in the eviction loop, so a buffer could be freed while still referenced by a later iteration — a textbook use-after-free introduced by a one-commit change. We confirmed against three more minidumps from the same bucket (all `texture.cpp:88`, all poison addresses), reverted that commit in a 4.2.1 hotfix, and watched the new signature's count flatten and then drop to zero over the next 24 hours as users updated. The numbers that made this fast: 11,400 reports → 7 buckets → 1 regression bucket of 6,200 → 1 commit → fixed. Bucketing turned an unreadable flood into a single, obvious, correlated decision. Without it, on-call would have read individual reports for a day.

## 9. War stories: famous crashes that a core would have explained

Post-mortem analysis is as old as crashes, and the field's hardest lessons are written in real incidents. A few worth knowing, told accurately.

**The Ariane 5 Flight 501 (1996).** Forty seconds after launch, the rocket veered, broke up, and self-destructed, destroying a payload worth hundreds of millions. The post-mortem — one of the most-studied in software history — traced it to an **uncaught operand-error exception** when a 64-bit floating-point horizontal-velocity value was converted to a 16-bit signed integer and overflowed. The value was larger on Ariane 5 than it had ever been on Ariane 4 (faster rocket), the conversion had no bound check because on Ariane 4 it provably could not overflow, and the unhandled exception shut down the inertial reference system — and its identical backup, which failed the same way a moment earlier. The diagnostic record (the avionics equivalent of a core dump and logs) is what let investigators reconstruct the exact conversion and exception. The lesson for us: a crash is the *symptom*; the captured state is what turns "it veered and exploded" into "an integer conversion overflowed at this instruction." The related class is in [integer overflow and floating-point traps](/blog/software-development/debugging/integer-overflow-and-floating-point-traps).

**Heartbleed (2014).** Not a crash but a *read overflow* — and a reminder of what cores show and what they cost. A missing bounds check in OpenSSL's TLS heartbeat let an attacker request more bytes back than they had sent, and the server happily returned adjacent heap memory — private keys, session data, whatever was next to the heartbeat buffer on the heap. The reason it leaked *secrets* specifically is the same allocator-reuse mechanism that makes use-after-free dangerous: freed and adjacent allocations sit next to each other on the heap, so reading past a buffer reads whatever the allocator placed nearby. A core dump of an exploited process would show exactly that adjacency — the heartbeat buffer and a private key in neighboring heap pages. It is also why you treat cores as **sensitive artifacts**: a full core contains the entire heap, including secrets, which is a real argument for minidumps in the field (stacks and registers, not the whole heap) and for tight access control on your core store.

**The crash that only happens at scale.** A realistic, illustrative composite that every backend engineer recognizes: a service runs fine for months, then starts crashing with `SIGABRT` from a failed assertion deep in a third-party library, but only on the busiest shard, only at peak, a few times a day, never reproducibly. Replaying traffic does nothing. With `LimitCORE=infinity` set, you capture a core from one crash, open it, run `thread apply all bt`, and find 200 threads — 199 of them blocked on the same mutex, and one inside the library's assertion having detected a corrupted internal structure. The corruption is a data race: under enough concurrency, two threads enter a supposedly-single-threaded library path and tear its state. You could never have caught it by reproduction; the core showed all 200 thread stacks at once, which is the one view that makes a contention bug obvious. This is the bridge to [race conditions, the hardest bugs to catch](/blog/software-development/debugging/race-conditions-the-hardest-bugs-to-catch) and [distributed race conditions and ordering](/blog/software-development/debugging/distributed-race-conditions-and-ordering): a core's all-thread snapshot is one of the few tools that catches a race after the fact.

The thread that runs through all three: **the crash is the cheap part; the captured state is the expensive, irreplaceable part.** Ariane's investigators, OpenSSL's responders, and the on-call engineer with 200 thread stacks all root-caused something un-reproducible because the state at the moment of failure was preserved and readable. That is the entire value proposition of this post.

## 10. How to reach for this (and when not to)

A core dump is the most complete debugging artifact there is, and that completeness is also its cost. Reach for post-mortem analysis with judgment.

**Reach for a core dump when** the crash does not reproduce, when it happened in production and you were not watching, when it is intermittent and rare, when it is multithreaded and you need every thread's state at once, or when it is a native segfault with no useful log. These are precisely the cases where nothing *but* a core can help, because every other technique requires the bug to happen again while you watch. If `ulimit -c unlimited` and a sane `core_pattern` are not set on your production hosts right now, that is the highest-leverage thing in this entire post — set them before you need them, because the crash you cannot reproduce already happened once and you only get the core if the plumbing was ready.

**Do not reach for a core dump when** a single log line would answer the question. If the crash is reproducible on demand, attach a live debugger and step — that is faster and richer than a frozen snapshot, because you can move forward in time. Do not `gcore` the payments process at peak unless you have to: snapshotting a multi-gigabyte process pauses it for the duration of the write, and on a latency-critical service that pause is a visible blip; prefer a replica or a canary. Do not collect full cores from the field — use minidumps, both for size and for privacy, because a full core is the user's entire heap including their secrets. Do not symbolize a core against "a build that's basically the same" — verify the build-id matches exactly or you will root-cause a fiction. Do not chase a heisenbug at `-O2` in the core if you can reproduce it at `-O0`: optimized cores have inlined frames and optimized-out locals (`<optimized out>` where a value should be), so a lower-optimization reproduction gives a far more readable corpse. And do not let cores pile up unbounded — a 4 GB core per crash will fill a disk and take *down* the host you were trying to debug, so cap the retention and the size.

The honest trade-off table:

| Technique | Best for | Cost / limitation |
| --- | --- | --- |
| Full core dump (gdb/dlv) | Non-reproducible native crash; all-thread state | Large (heap-sized); contains secrets; static |
| Heap dump + MAT | JVM/.NET OOM and memory leaks | Heap-sized; managed objects only; not native frames |
| `gcore` live snapshot | Hung/leaking process you must not kill | Pauses the process during the write |
| Minidump (Breakpad/Crashpad) | Field crashes at scale; privacy-sensitive | No full heap; no retained-size leak analysis |
| Live debugger (attach) | Reproducible bug; need to step forward | Requires reproduction; intrusive in prod |
| Logs / metrics / traces | Knowing *that* and *where* it failed | Rarely enough to know *why* at the byte level |

The pattern across the table: cores and heap dumps answer "*why* did this specific death happen," logs and traces answer "*that* and *where* it happened," and a live debugger answers "what happens *next* as I step." A mature debugging practice uses all three — traces to find the crashed instance, the core to root-cause it, and a live debugger to confirm the fix reproduces and is gone. The core dump is the irreplaceable middle: it is the only one that works *after the fact* on a bug that will not come back.

## Key takeaways

- **A core dump is a frozen debugger session.** Registers, every thread's stack, locals, and the heap, captured at the instant of death — you can inspect the dead process exactly as if you had been attached, on another machine, days later, without reproducing anything.
- **Make cores exist before you need them.** `ulimit -c unlimited` (or `LimitCORE=infinity`), a sane `/proc/sys/kernel/core_pattern` or `coredumpctl`, and the per-runtime switch (`GOTRACEBACK=crash`, `-XX:+HeapDumpOnOutOfMemoryError`, `dotnet-dump collect`, `faulthandler`). A missing core is a configuration bug, not bad luck.
- **The faulting address tells you the bug class for free.** `0x0` is a null dereference; a small offset like `0x18` is null-plus-a-field; a freed-poison pattern like `0xdeadbeef` is a use-after-free; ASCII bytes in a pointer slot are a buffer overflow.
- **Symbolize against the exact build-id.** A stripped binary gives `?? ()`; separate `.debug`/dSYM/PDB files keyed by build-id turn addresses into file and line. Matching the wrong build root-causes a fiction — verify the build-id before you trust a frame.
- **The autopsy is a fixed sequence.** Open the core, `thread apply all bt` to find the crashed thread, `bt full` to read locals, `frame N` into your code, `info args`/`info locals`/`print`/`x/` to read the bad value. Same shape in `dlv core`, MAT, and `dotnet-dump`.
- **For OOM and leaks, the dominator tree is the answer.** Retained size, not shallow size, names the culprit; "path to GC roots" tells you why it cannot be collected. The histogram lies — it shows leaf arrays, never the map that holds them.
- **Containers share the host `core_pattern`.** A pod's core is decided by the node, not the container; collect via a node-level path plus a sidecar, or an in-process minidump handler. The field needs minidumps for size and privacy, not full cores.
- **Bucket crashes by stack signature.** Normalize the top frames, hash them, and ten thousand reports become five bugs ranked by frequency × severity; a new bucket after a deploy is a visible regression and your first bisection hypothesis.
- **Treat cores as sensitive.** A full core contains the entire heap, secrets included; control access, prefer minidumps in the field, and cap retention so a 4 GB-per-crash pile does not take down the host.

## Further reading

- [Debugging with GDB: The GNU Source-Level Debugger](https://sourceware.org/gdb/current/onlinedocs/gdb/) — the canonical manual; the chapters on core files, `thread apply`, and examining memory are the reference for everything in section 4.
- [`core(5)` and `core_pattern` in the Linux man-pages](https://man7.org/linux/man-pages/man5/core.5.html) — the authoritative description of when cores are written, the size limit, and every `%` specifier in `core_pattern`.
- [`coredumpctl` and `systemd-coredump` documentation](https://www.freedesktop.org/software/systemd/man/coredumpctl.html) — how modern Linux captures, stores, and serves cores, and how to extract one for offline analysis.
- [Eclipse Memory Analyzer (MAT) documentation](https://eclipse.dev/mat/) — the dominator-tree and leak-suspects analyses that turn a 4 GB heap dump into a one-object culprit, with the retained-size concept explained in depth.
- [The Mozilla / Google Breakpad and Crashpad projects](https://chromium.googlesource.com/breakpad/breakpad/) — how field crash reporting, minidumps, and server-side symbolization actually work at scale.
- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe→reproduce→hypothesize→bisect→fix→prevent loop this post applies when *reproduce* is impossible and the only witness is a corpse.
- [Reading a stack trace across languages](/blog/software-development/debugging/reading-a-stack-trace-across-languages) — the frame-reading skill that the core dump supercharges by adding every local value to the call chain.
- [Mastering an interactive debugger](/blog/software-development/debugging/mastering-an-interactive-debugger) and [hunting memory leaks and bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) — the live-debugger and live-leak counterparts to the post-mortem techniques here; the dominator-tree reasoning is shared.
