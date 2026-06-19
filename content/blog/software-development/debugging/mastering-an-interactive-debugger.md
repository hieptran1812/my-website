---
title: "Mastering an Interactive Debugger: Attach, Post-Mortem, Script, and Rewind"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Go past breakpoints into the debugger features most engineers never touch — attaching to a wedged production process and dumping every thread stack, opening a core dump, scripting gdb to auto-print and continue, and reverse-stepping with rr from a segfault back to the write that planted the dangling pointer, in gdb, lldb, pdb, ipdb, and delve."
tags:
  [
    "debugging",
    "software-engineering",
    "gdb",
    "lldb",
    "pdb",
    "delve",
    "rr",
    "core-dump",
    "reverse-debugging",
    "remote-debugging",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/mastering-an-interactive-debugger-1.png"
---

It is 3:14 in the morning and the checkout service for a payments platform has stopped responding. Requests pile up. Health checks go red. The on-call playbook says, in bold, "if unresponsive, restart the pod." Your finger hovers over the button. And here is the trap: the moment you restart, the bug is gone. Not fixed — gone. The exact arrangement of held locks, half-finished goroutines, and a thread blocked forever on a mutex that another thread will never release evaporates the instant the process dies. Tomorrow at 2pm it will wedge again, and you will know nothing more than you do right now.

There is another door. Instead of killing the process, you walk up to it while it is still alive and pin it under a microscope. `gdb -p 8421`. `dlv attach 8421`. Thirty seconds later you have every thread's stack dumped to your terminal, you can see thread 7 holding lock A and waiting on lock B while thread 12 holds lock B and waits on lock A, and you have the root cause — a classic lock-order inversion — without restarting anything and without losing a single byte of evidence. Then you detach, restart to clear the incident, and ship the fix you actually understand.

This post is the deep companion to the intro that treats [the debugger as a microscope](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it). That piece taught you to break, step, and inspect — the 5% of the tool everyone uses. This piece is the other 95%: the power features that turn a debugger from "a fancier print statement" into a forensic instrument. We will cover four doors onto a program that you almost never need to restart, shown in Figure 1: **attaching** to a process that is already running (including one buried inside a container or Kubernetes pod), **post-mortem** debugging of a core dump after the process is already dead, **scripting** the debugger so it runs a whole investigation unattended, and **reverse / replay debugging** where you literally run the program backward in time from a crash to its cause. We will do all of it in `gdb` and `lldb` for native code, `pdb` and `ipdb` for Python, and `delve` for Go, with real, copy-able sessions.

![A flow diagram showing a live wedged process, a crashed process, and a recorded trace each feeding into attach, post-mortem, and replay entry doors that all arrive at the exact root-cause line without a restart](/imgs/blogs/mastering-an-interactive-debugger-1.png)

By the end you will be able to: attach to a hung production process and read the deadlock straight off the thread stacks; open a core dump and stand exactly where the program died; write a `.gdbinit` and breakpoint command lists that automate an entire repetitive investigation; and use `rr` to reverse-execute from a `SIGSEGV` back to the line that stored the bad pointer thousands of lines earlier. This is squarely the **observe → reproduce → hypothesize → bisect → fix → prevent** loop from [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), with the debugger doing the heavy lifting of *observe* and *bisect* — except now we are binary-searching through time and across threads, not just down a call stack.

## 1. The mental shift: a debugger is a process you can borrow

Before the power features, one idea has to land, because everything else follows from it. A debugger is not a special mode you compile your program into. It is a separate process that borrows control of *another* process through an operating-system facility. On Linux that facility is the `ptrace` system call. When you run `gdb -p 8421`, gdb calls `ptrace(PTRACE_ATTACH, 8421, ...)`, the kernel stops process 8421 in its tracks, and from that moment gdb can read and write 8421's memory and registers, single-step its instructions, and read its thread states — all from the outside.

That is the whole trick, and it is why attaching works on a process you did not launch under the debugger, why a core dump (a frozen snapshot of that same memory) can be debugged offline, and why `rr` can record every non-deterministic input and replay it deterministically. The reason this matters for your 3am incident is concrete: **you do not have to have started the process under gdb to debug it.** Production processes are started by systemd, by a container runtime, by Kubernetes. You attach to them where they live.

There is a cost, and you must respect it. `PTRACE_ATTACH` *stops* the target. For the duration that gdb has it stopped — while you type commands, while you print structures — the target is frozen. Attach to your payments process and forget to `continue`, and every in-flight request hangs. So the discipline is: attach, grab what you need fast (often a single `thread apply all bt`), and detach. We will come back to this when we talk about what *not* to do.

One operational detail bites people the first time: on many modern Linux systems, `ptrace` of a process you do not own is restricted by the `kernel.yama.ptrace_scope` sysctl. With the default value of 1, a process can only attach to its own direct children, so `gdb -p` of an unrelated process fails with "Operation not permitted" even as your own user. The fixes, in order of preference, are: run the debugger as root (`sudo gdb -p`), which always works; grant the `CAP_SYS_PTRACE` capability to the debugger; or, as a last resort on a development box, loosen the sysctl with `echo 0 | sudo tee /proc/sys/kernel/yama/ptrace_scope`. In a container, the same restriction shows up as a missing `SYS_PTRACE` capability — which is exactly why `kubectl debug` and `docker run --cap-add=SYS_PTRACE` from Section 2.2 exist. Knowing this turns a baffling permission error into a one-line fix instead of a half-hour of confusion.

One more piece of vocabulary, because the rest of the post leans on it. A **thread** is an independent flow of execution sharing the process's memory. A **stack** (or call stack, or backtrace) is the ordered list of function calls that led a thread to where it currently is — frame `#0` is the innermost (where it is right now), and the numbers climb toward `main`. A **mutex** or **lock** is an object exactly one thread can hold at a time; others that ask for it block until it is released. A **deadlock** is when a set of threads each hold a lock another needs, forming a cycle, so none can ever proceed. Hold those four, and the deadlock investigation below reads itself.

## 2. Attaching to a running process

Attaching is the single most underused debugger skill, and it is the one that pays off in production. The symptom that screams "attach" is a process that is *alive but wrong*: hung, spinning at 100% CPU, leaking memory, or producing bad output, but not crashing. A crash gives you a core dump (Section 3); a hang gives you nothing — unless you go in and take it.

### 2.1 The base case: gdb, lldb, delve, pdb

In native code, attaching is one flag. Find the PID with `pgrep` or `ps`, then:

```bash
# Native C/C++: attach gdb to a running process by PID
$ pgrep -a checkout
8421 /usr/local/bin/checkout --config /etc/checkout.yaml
$ sudo gdb -p 8421
# ... gdb stops the process and drops you at a prompt ...
(gdb) bt                 # backtrace of the current thread
(gdb) info threads       # list every thread and where it is
(gdb) thread apply all bt # the money command: every thread's stack
(gdb) detach             # let the process run again
(gdb) quit
```

`lldb` is identical in spirit — `lldb -p 8421` or, inside lldb, `process attach --pid 8421` (or `--name checkout` to attach by name). The command to dump all threads is `bt all`.

Go programs use `delve`. Go's runtime multiplexes thousands of goroutines onto a handful of OS threads, so the equivalent of "all thread stacks" is "all goroutine stacks":

```bash
# Go: attach delve to a running process and dump every goroutine
$ sudo dlv attach 8421
(dlv) goroutines          # list all goroutines, one line each
(dlv) goroutines -t       # ... with full stack traces (the money command)
(dlv) goroutine 137       # switch to a specific goroutine
(dlv) stack               # its stack
(dlv) detach              # leave the process running (no -k means don't kill)
```

Be careful with `dlv`: `dlv attach 8421` followed by `quit` will ask whether to kill the process. Answer no, or use `detach` explicitly. Killing the production process you were trying to *not* restart is an embarrassing way to end an incident.

Python is the interesting case because CPython does not expose `ptrace`-style attach natively — the interpreter has to cooperate. There are three ways in, of increasing intrusiveness, and you should know all three:

```bash
# 1. py-spy: zero-cooperation, read-only snapshot of all thread stacks
$ sudo py-spy dump --pid 8421
# Prints every Python thread's stack without stopping the process for
# more than a few milliseconds. SAFEST in prod — no code changes, no real freeze.

# 2. debugpy: an interactive pdb-like session over a socket, if you started
#    the process listening (or can import and start the listener)
#    In the app:  import debugpy; debugpy.listen(("0.0.0.0", 5678))
$ python -m debugpy --listen 0.0.0.0:5678 --wait-for-client app.py
# then connect VS Code or any DAP client to port 5678.

# 3. remote-pdb: drop a breakpoint that opens a telnet-able pdb
#    In the app:  from remote_pdb import set_trace; set_trace(host="0.0.0.0", port=4444)
$ nc 127.0.0.1 4444   # you now have a full pdb prompt on the live process
```

The hierarchy is worth internalizing because it maps onto the cost. `py-spy dump` is read-only and barely touches the process; reach for it first in production to answer "where is it stuck?" `debugpy` and `remote-pdb` give you a *real* interactive prompt where you can evaluate expressions and step — far more powerful, but they require the process to be cooperating (listening on a socket), and they fully stop it when you break. The way this works mirrors gdb: a separate client drives the interpreter through a control channel.

There is a fourth Python option worth knowing for the truly stuck case, where the process is not cooperating and you need *interactive* access, not just a snapshot: you can use gdb itself with the CPython extensions. CPython ships a `python-gdb.py` script (and most distributions install it) that adds `py-bt`, `py-list`, and `py-locals` commands to gdb, so `gdb -p 8421` followed by `py-bt` gives you the *Python-level* backtrace of a C-level process, even when the interpreter is wedged in a C extension or holding the GIL. It is the bridge between the native and the Python worlds: gdb sees the C stack, and `py-bt` reconstructs the Python frames sitting on top of it from the interpreter's internal data structures. This is how you debug a Python process that has deadlocked inside a native library — the pure-Python tools cannot get a word in, but gdb can attach to anything.

The practical decision among the four comes down to two questions: *can I change or restart the process?* and *do I need a snapshot or an interactive session?* If you can restart, launch under `debugpy --listen --wait-for-client` and you have a clean interactive session from the start. If you cannot touch it and only need to know where it is stuck, `py-spy dump`. If you cannot touch it and need to interactively poke at a wedged native extension, `gdb -p` plus `py-bt`. If you control the code and want a break point you can telnet into on demand, `remote-pdb`. Four doors, one decision tree.

#### Worked example: a wedged production process, no restart

A Go checkout service stops serving. CPU is near zero — it is not spinning, it is *blocked*. The pod has been wedged for six minutes and the playbook says restart. Instead:

```bash
$ kubectl exec -it checkout-7d9f-abc12 -- sh
/ # ps aux | grep checkout
   1 root  /usr/local/bin/checkout
/ # dlv attach 1
(dlv) goroutines -t 2>&1 | head -60
Goroutine 18 - User: order.go:142 (*Ledger).Debit
        sync.runtime_SemacquireMutex
        sync.(*Mutex).Lock
        order.go:142 (*Ledger).Debit          <- waiting on accountMu
Goroutine 24 - User: order.go:97 (*Ledger).Credit
        sync.runtime_SemacquireMutex
        sync.(*Mutex).Lock
        order.go:97 (*Ledger).Credit          <- waiting on ledgerMu
```

Two goroutines, both parked in `sync.(*Mutex).Lock`. Goroutine 18 is in `Debit`, blocked acquiring `accountMu`; goroutine 24 is in `Credit`, blocked acquiring `ledgerMu`. Read the code at those two lines and the picture completes: `Debit` takes `ledgerMu` then `accountMu`; `Credit` takes `accountMu` then `ledgerMu`. Opposite lock order. Goroutine 18 holds `ledgerMu` and wants `accountMu`; goroutine 24 holds `accountMu` and wants `ledgerMu`. That is a cycle. That is a deadlock, and you found it in under two minutes without a restart. We will return to *why* this is a cycle, and how to read it off the stacks mechanically, in Section 7. The fix is to make both paths acquire the locks in the same global order, which we will also cover.

For the broader class of hang — deadlocks, livelocks, lock-order inversions — there is a planned sibling post, `deadlocks-livelocks-and-starvation`, that goes deep on the concurrency theory. Here we stay focused on the debugger move: **attach, dump all stacks, find the cycle.**

### 2.2 Attaching inside a container or pod

The 2026 reality is that your wedged process lives inside a container, and the container does not have gdb, delve, or even `ps`. There are two clean ways in, and one thing you must get right: the debugger and the target must share a **PID namespace**, or the debugger cannot see the target's process ID, and `ptrace` will fail.

The first way, if your container image happens to include the tools, is to exec into it (`kubectl exec -it POD -- sh`) and run the debugger there, as in the worked example above. That is the simplest path when it works.

The second way — the right one for slim production images — is `kubectl debug`, which attaches a brand-new *ephemeral container* (with all your tools) into the running pod, sharing the target's process namespace:

```bash
# Attach an ephemeral debug container that shares the target's PID namespace
$ kubectl debug -it checkout-7d9f-abc12 \
    --image=ghcr.io/yourorg/debug-tools:latest \
    --target=checkout \
    --share-processes
# Inside the ephemeral container you can now SEE the checkout process:
/ # ps aux
PID   USER  COMMAND
1     root  /usr/local/bin/checkout      <- the target, visible because
                                            --target shares its PID namespace
/ # dlv attach 1
```

The `--target=checkout` flag is the load-bearing part: it makes the ephemeral container share the PID (and other) namespaces of the named container, so the target's PID is visible. Without it you would only see your own debug container's processes. You also need `SYS_PTRACE` capability available; for raw Docker that is `docker run --cap-add=SYS_PTRACE` (or `--pid=container:NAME` to share the namespace from outside), and for Kubernetes the ephemeral debug container needs the capability in clusters that restrict it. The general principle — sharing a namespace to reach across an isolation boundary — is the same idea that underpins a lot of container-debugging in [debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale).

### 2.3 The hung process, more deeply: dumping all thread stacks

The single command that pays for this whole section is "dump every thread's stack." It is the first thing to run on any hang, in any language, because a hang is, by definition, threads stuck somewhere, and the stack tells you *where* and *waiting on what*.

Here is the full vocabulary across the toolchain:

```bash
# gdb (C/C++/Rust):
(gdb) thread apply all bt            # every thread, full backtrace
(gdb) thread apply all bt full       # ... plus local variables in each frame

# lldb:
(lldb) thread backtrace all          # or:  bt all

# delve (Go):
(dlv) goroutines -t                  # every goroutine with stack

# Python (no debugger needed for a snapshot):
$ py-spy dump --pid 8421             # external, read-only
# or, from inside the process, faulthandler dumps all Python thread stacks:
#   import faulthandler, signal
#   faulthandler.register(signal.SIGUSR1)   # then:  kill -USR1 8421

# JVM (for completeness — same idea, different tool):
$ jstack 8421                        # every Java thread's stack + lock info
```

That `faulthandler.register(signal.SIGUSR1)` line is a gem. Add it at startup and you can, at any time, `kill -USR1 <pid>` to make a *hung* Python process print every thread's stack to stderr — no attach, no restart, works even when the GIL-holding thread is wedged. It is the Python equivalent of `jstack`, and it costs you three lines at startup.

What you are looking for in the dump is any thread parked in a *blocking* call — `pthread_mutex_lock`, `sync.(*Mutex).Lock`, `futex`, `recv`, `read`, `epoll_wait`, `__lll_lock_wait`. A thread blocked in `epoll_wait` is fine, that is an idle event loop waiting for work. A thread blocked in `pthread_mutex_lock` while *another* thread holds that mutex and is itself blocked is the deadlock. The skill is reading the dump and asking, for each blocked thread, "what is it waiting for, and who holds that?"

A second pattern the all-stacks dump catches, and one that fools people because it does not *look* like a hang at first, is the **thread pool exhaustion** deadlock. Picture a pool of, say, 16 worker threads. Each worker, while processing a request, makes a call that itself needs to be serviced by another worker from the *same* pool — and waits for it. If all 16 workers are busy waiting on tasks that can only be completed by a worker, none of which is free, the pool is wedged with zero CPU and no error, exactly like a lock deadlock. Dump all 16 stacks and you will see the tell: every one of them is parked in the same "wait for sub-task" call, and the pool's work queue has entries no one will ever pick up. The fix is structural (never block a pool thread on another task from the same pool; use a separate pool, or make the dependency async), but you cannot *see* the structural problem until the all-stacks dump shows you sixteen identical waiting stacks. This same shape causes the classic file-descriptor-leak-into-pool-starvation cascade and the database connection pool exhaustion that shows up in [database locks and deadlocks in production](/blog/software-development/database/database-locks-and-deadlocks-in-production) — the symptom is a hang, the dump shows uniform waiting, and the root cause is a resource that was acquired but never returned.

How do you know which thread *holds* a contended lock, not just which ones are waiting for it? In gdb with full debug info, `thread apply all bt full` prints locals, and a `pthread_mutex_t`'s internal `__owner` field holds the thread ID of the current owner — `print mutex.__data.__owner` gives you the holder's kernel TID directly, and `info threads` maps that TID back to a gdb thread number. In Go, `dlv` shows the mutex state in the goroutine that holds it because the holding goroutine is *not* parked in `Lock` — it is off doing other work with the lock held, so you find it by elimination (it is the goroutine touching the protected data that is not itself blocked). In the JVM, `jstack` is the friendliest of all: it prints `- locked <0x...>` and `- waiting to lock <0x...>` annotations directly on each frame, and modern `jstack` even prints "Found one Java-level deadlock" with the full cycle spelled out. Different tools, same question: who holds what, who waits for what, and is there a cycle.

## 3. Post-mortem: debugging the dead

Sometimes the process does not hang — it dies. A segfault, an abort, an unhandled exception, an OOM kill. The process is gone before you can attach. The answer is the **core dump**: a file containing a complete snapshot of the process's memory, registers, and thread states at the moment it died. Open it in a debugger and you stand exactly where the crash happened, can walk the stack, and can inspect every variable — as if you had a breakpoint on death itself.

This connects directly to the planned Track-E sibling `reading-a-core-dump-post-mortem-analysis`, which goes deep on reading core dumps in anger. Here, the debugger angle: how to *generate* a core on demand and how to *open* one.

### 3.1 Getting a core in the first place

By default, most Linux systems disable core dumps (the size limit is zero). You have to enable them:

```bash
# Allow unlimited core dumps for processes you launch from this shell
$ ulimit -c unlimited
$ ./buggy            # if it segfaults, you get a 'core' file

# On systemd systems, cores are captured by coredumpctl regardless of ulimit
# in the shell — they go to the systemd-coredump journal:
$ coredumpctl list                 # show recent crashes
$ coredumpctl info buggy           # summary, signal, backtrace
$ coredumpctl gdb buggy            # open the most recent core in gdb directly
```

But the powerful move is generating a core *on demand* from a process that has not crashed — for a hang, or to snapshot a process for offline study without keeping production stopped:

```bash
# gcore: dump a core of a LIVE process without killing it
$ sudo gcore 8421
Saved corefile core.8421
# The process keeps running; you now study core.8421 at your leisure,
# offline, on your laptop, while prod stays up.
```

`gcore` is the under-appreciated hero of production debugging: it captures the frozen evidence (like attaching) but then *lets the process go*, so you can investigate without holding prod hostage. Attach freezes prod for as long as you poke at it; `gcore` freezes it for the few hundred milliseconds it takes to write the dump, then you analyze the file. Figure 8 contrasts this "capture then study" posture against the "restart and hope" reflex.

For Go, set `GOTRACEBACK=crash` in the environment and the runtime will dump a core (and print all goroutine stacks) on an unrecoverable panic instead of just exiting:

```bash
$ GOTRACEBACK=crash GORACE="halt_on_error=1" ./service
# On panic: prints every goroutine's stack AND drops a core for `dlv core`.
```

For Python, `faulthandler` is the equivalent for the *crash* case — it installs handlers so that a fatal error (segfault in a C extension, for example) dumps the Python stack instead of dying silently:

```python
# Put this at the top of your entrypoint. Now a hard crash in a C
# extension prints the Python traceback to stderr before the process dies.
import faulthandler
faulthandler.enable()
```

### 3.2 Opening a core dump

Once you have a core, opening it is one line. The crucial requirement: you need the *same binary* and ideally its **debug symbols** (the table mapping addresses back to function names, line numbers, and variable names). Strip the symbols and your backtrace is a list of hex addresses; keep them and it reads like source.

```bash
# Native: open the program together with its core
$ gdb ./buggy core.8421
(gdb) bt                  # the stack at the moment of death
(gdb) thread apply all bt # all threads at death
(gdb) frame 3             # jump to frame #3
(gdb) info locals         # local variables in that frame
(gdb) print some_struct   # inspect any value, exactly as it was at death
(gdb) print $_siginfo     # what signal killed it, and the faulting address

# lldb equivalent:
$ lldb ./buggy -c core.8421
(lldb) bt all

# Go:
$ dlv core ./service core.8421
(dlv) bt
(dlv) goroutines -t
(dlv) frame 2
(dlv) locals
```

The first command to run on any core is `bt` (and `thread apply all bt`), then look at frame `#0` — that is where it died. Then `print $_siginfo` (gdb) or read the signal from `coredumpctl info` to learn *which* fault it was: `SIGSEGV` with a fault address of `0x0` is a null dereference; a fault address of garbage like `0x7f...beef` is usually a use-after-free or a wild pointer; `SIGABRT` is an assertion or a `glibc` heap-corruption abort.

### 3.3 Python post-mortem without a core

Python has its own, lovely post-mortem mode that needs no core file at all. When an exception propagates to the top and crashes your script, `pdb.pm()` drops you into a debugger *at the frame where the exception was raised*, with the entire stack still inspectable:

```python
>>> import pdb
>>> try:
...     run_the_thing()
... except Exception:
...     pdb.post_mortem()      # debug at the point the exception was raised
```

Or, even better for interactive work, launch the whole script under post-mortem control so *any* uncaught exception drops you into pdb automatically:

```bash
# -m pdb runs the script; on an unhandled exception it enters post-mortem
$ python -m pdb -c continue app.py
# When app.py raises and would have crashed, you land in pdb at the raising
# frame. Now: `up`, `down`, `p variable`, `args`, `list` — full forensic access.

# IPython users get the same with the magic:
In [1]: %pdb on        # auto-post-mortem on every exception
In [2]: %debug         # post-mortem the LAST exception, after the fact
```

`ipdb` (the IPython-flavored pdb) makes this nicer with tab completion and syntax highlighting, and `%debug` is the single most useful thing in an interactive session: an exception just flew by, you type `%debug`, and you are standing at the crash site with the full stack. The key insight is that an exception traceback is a *post-mortem* — Python preserves the frames, and `pdb.pm()` / `%debug` lets you walk them.

#### Worked example: the null deref that crashed 10k lines later

A C++ service segfaults in production roughly once an hour. No pattern in the logs. With `ulimit -c unlimited` set under systemd, `coredumpctl` has the cores:

```bash
$ coredumpctl gdb svc          # opens the latest core
(gdb) bt
#0  0x0000556... in Session::flush() at session.cc:88
#1  0x0000556... in Worker::tick() at worker.cc:204
...
(gdb) frame 0
(gdb) print this
$1 = (Session *) 0x0           # `this` is null — we called a method on a
                               # destroyed/never-constructed Session
(gdb) print $_siginfo.si_addr
$2 = (void *) 0x18             # faulted reading offset 0x18 of a null `this`
```

`this == 0x0` and the fault address is a small offset (`0x18`) — that is the unmistakable signature of calling a method through a null pointer and dereferencing a member. The crash is at `session.cc:88`, but the *bug* is wherever a `Session*` got nulled or freed. The core localizes the crash; finding the cause across those 10k lines is exactly the job for reverse debugging in Section 5. Before the core, you had "segfault, once an hour, no idea." After it: "null `this` in `Session::flush`, called from `Worker::tick`." That is the whole value of post-mortem — it converts a vague, intermittent crash into a specific, reproducible question.

## 4. Scripting the debugger: turn an investigation into a program

Here is where most engineers stop, and where the real leverage is. A debugger is not just an interactive prompt — it is *programmable*. You can attach a list of commands to a breakpoint, conditionally fire, auto-print and continue, and drive an entire investigation from a script with zero manual stepping. The difference is the difference between Figure 5's two columns: pressing "continue" 3.8 million times by hand versus one unattended run that logs exactly what you asked for.

![A before-and-after comparison contrasting manual stepping that repeats millions of times by hand against a dprintf and breakpoint command list that auto-prints and continues in a single unattended run](/imgs/blogs/mastering-an-interactive-debugger-5.png)

### 4.1 Breakpoint command lists: scripted printf

The foundational trick: attach commands to a breakpoint so that when it hits, gdb runs them and (optionally) continues automatically. This is "scripted printf" — you get logging from inside the debugger without recompiling, without editing source, on a binary you cannot rebuild.

```bash
# gdb: break, attach a command list that prints and continues
(gdb) break allocate_block
(gdb) commands
> silent                          # don't announce the stop
> printf "alloc size=%d caller=%p\n", size, __builtin_return_address(0)
> continue                        # auto-resume; no manual interaction
> end
(gdb) run
# Now every call to allocate_block prints one line and keeps going.
# You just instrumented a function with NO source change.
```

`dprintf` is the same thing in one line — a "dynamic printf" breakpoint that prints and continues by construction:

```bash
# gdb dprintf: a breakpoint that is purely a print-and-continue
(gdb) dprintf order.cc:142, "Debit acct=%d amount=%d\n", acct_id, amount
(gdb) run
# Equivalent to inserting a printf at order.cc:142 without touching the code.
```

Conditional and counting variants make this surgical. A **conditional breakpoint** fires only when a predicate is true; an **ignore count** skips the first N hits. These are how you stop on iteration 3,847,221 of a loop without pressing continue 3.8 million times:

```bash
(gdb) break process_record if record->id == 0x8badf00d   # only the bad one
(gdb) break inner_loop
(gdb) ignore 2 3847220        # skip 3,847,220 hits of breakpoint 2, stop on the next
```

pdb and delve have the same capabilities:

```python
# pdb: a conditional breakpoint and a command list
(Pdb) break payments.py:88, amount < 0      # only when amount goes negative
(Pdb) commands 1
(com) p f"neg amount {amount} for {acct}"
(com) continue
(com) end
```

```bash
# delve: a tracepoint (breakpoint that prints and continues) + on-hit commands
(dlv) trace order.go:142                     # like dprintf for Go
(dlv) break order.go:142
(dlv) on 1 print amount                      # run `print amount` each hit
(dlv) condition 1 amount < 0                 # only when amount < 0
```

#### Worked example: the conditional breakpoint that fired on iteration 3,847,221

A hashing routine corrupts one record in a batch of four million. You suspect a specific key collision but stepping through four million iterations is absurd. You set a conditional breakpoint on the symptom — the moment a bucket's count goes negative, which should be impossible:

```bash
(gdb) break bucket.c:51 if bucket->count < 0
(gdb) run
# ... runs at full speed for ~3 seconds ...
Breakpoint 1, insert at bucket.c:51
(gdb) print iteration
$1 = 3847221
(gdb) print *bucket
$2 = {key = 0x0, count = -1, next = 0x6261...}   # key is null, count went -1
```

The breakpoint fired exactly once, on iteration 3,847,221, at the instant the invariant broke — `count` went negative because a null key hashed to a bucket that had been concurrently resized. No stepping, no millions of continues; the *condition* did the searching. This is binary search in disguise: instead of bisecting commits or stack frames, you bisect *iterations of a loop* by predicating the stop on the failure condition itself. The cost is real — a conditional breakpoint evaluates its predicate every hit, so this run was slower than native, but seconds versus the hours manual stepping would take.

### 4.2 .gdbinit, the Python API, and automating a whole investigation

gdb embeds a full Python interpreter. You can define commands, write pretty-printers, walk data structures, and automate analyses that would be tedious by hand. Put reusable setup in `~/.gdbinit` (loaded on every start) or a project `.gdbinit`, and one-off automations in a script you run with `gdb -x script.gdb` or `source script.py`.

```python
# A gdb Python script (source it with: (gdb) source dump_threads.py)
# Walk every thread and print the top frame's function — a custom "where".
import gdb

class WhereAll(gdb.Command):
    def __init__(self):
        super().__init__("where-all", gdb.COMMAND_USER)
    def invoke(self, arg, from_tty):
        for thread in gdb.selected_inferior().threads():
            thread.switch()
            frame = gdb.newest_frame()
            print(f"thread {thread.num}: {frame.name()} "
                  f"at {frame.find_sal().symtab.filename}:"
                  f"{frame.find_sal().line}")

WhereAll()   # registers the `where-all` command
```

You can also drive a complete, unattended investigation as a batch script — no human at the keyboard. This is how you let a flaky test run a thousand times under the debugger overnight and capture the state only on the failing run:

```bash
# investigation.gdb — runs start to finish with no interaction
set pagination off
break corrupt_invariant if state->magic != 0xCAFEBABE
run
# When (if) the breakpoint hits, dump everything and quit:
thread apply all bt full
printf "=== faulting state ===\n"
print *state
generate-core-file /tmp/failure.core   # save a core for later study
quit
```

```bash
# Run it headless against the test, retrying until it fails:
$ for i in $(seq 1 1000); do
>   gdb -batch -x investigation.gdb --args ./flaky_test || break
> done
# Most runs sail past (breakpoint never hits, program exits 0). The one run
# that corrupts the invariant stops, dumps all stacks + a core, and the loop
# stops. You wake up to /tmp/failure.core sitting exactly at the failure.
```

That pattern — a programmable breakpoint on the *invariant*, run under a repeat-until-fail loop, capturing a core only on the bad run — is one of the highest-leverage techniques in this whole post. It marries the debugger to the reproduce-it-first discipline from [reproduce it first or you are not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging): you do not have to be watching when the heisenbug strikes; the script catches it for you.

A subtler form of the same idea is the **logging breakpoint that builds a history you diff later**. Instead of stopping on a condition you already know, you `dprintf` a tracepoint at a few suspicious sites, run the program once when it behaves and once when it misbehaves, and *diff the two logs*. The first line where the logs diverge is the boundary between correct and incorrect behavior — a bisection in the dataflow, not in the source history. This is the same binary-search instinct as [bisecting a regression](/blog/software-development/debugging/binary-search-your-bug-with-bisection), applied to a single run's execution trace rather than to a sequence of commits. The debugger script is the instrument; the diff is the bisection.

For a fully automated investigation, gdb's batch mode plus the Python API lets you express the *whole hypothesis* as code. Suppose you suspect a counter is being decremented below zero somewhere, but you do not know where. You can register a Python breakpoint that fires on every write to the counter's function, inspects the value, and only reports when it crosses zero:

```python
# autobisect.py — sourced into gdb, finds the first decrement-below-zero.
import gdb

class CounterWatch(gdb.Breakpoint):
    def __init__(self):
        super().__init__("decrement_counter")
    def stop(self):                          # return True to actually stop
        val = int(gdb.parse_and_eval("counter"))
        if val <= 0:
            print(f"counter hit {val} at:")
            gdb.execute("bt")
            return True                      # stop here — we found it
        return False                         # keep running, don't bother me

CounterWatch()
gdb.execute("run")
```

`gdb -batch -x autobisect.py --args ./prog` runs unattended and stops only at the precise moment the invariant breaks, with a backtrace already printed. The `stop()` method returning `False` is the key: gdb evaluates it on every hit but only surfaces the one that matters, so you get the speed of "run to the failure" with the precision of "inspect every candidate." This is what it means to turn an investigation into a program.

### 4.3 Pretty-printers for your own types

By default gdb prints your structs as raw fields, which is useless for a custom container — a `std::unordered_map` prints as a tangle of bucket pointers, your ring buffer as an opaque array. A **pretty-printer** is a small Python class that teaches gdb how to display *your* type readably. Write one once and every `print` of that type, in every session and every core dump, is legible.

```python
# A gdb pretty-printer for a custom RingBuffer<T>.
# Register it and `print rb` shows the live elements, not the raw array.
import gdb.printing

class RingBufferPrinter:
    def __init__(self, val):
        self.val = val
    def to_string(self):
        head = int(self.val['head'])
        size = int(self.val['size'])
        cap  = int(self.val['capacity'])
        buf  = self.val['data']
        items = [str(buf[(head + i) % cap]) for i in range(size)]
        return f"RingBuffer(size={size}/{cap}, [{', '.join(items)}])"

def build_pp():
    pp = gdb.printing.RegexpCollectionPrettyPrinter("mylib")
    pp.add_printer('RingBuffer', '^RingBuffer<.*>$', RingBufferPrinter)
    return pp

gdb.printing.register_pretty_printer(gdb.current_objfile(), build_pp())
```

The payoff is enormous on a core dump: instead of squinting at `head`, `tail`, and a raw array to reconstruct what was in the buffer when it crashed, `print rb` just shows you the elements in order. The C++ standard library ships printers like this for `std::` types (that is why `std::vector` prints nicely); writing them for *your* types is a few lines and pays back every time you debug that code. lldb has the same concept via "synthetic children" and Python type summaries.

### 4.4 Convenience variables and computed walks

gdb has **convenience variables** — names starting with `$` that you assign to and reuse — and they turn the prompt into a small programming environment for walking data structures. This is how you traverse a linked list looking for the corrupted node without writing a hundred manual `print` commands:

```bash
# Walk a linked list in gdb until you find the bad node, using a
# convenience variable as the cursor and a while loop.
(gdb) set $node = head
(gdb) while $node != 0
 >  if $node->checksum != $node->expected
 >    printf "corrupt node at %p, value=%d\n", $node, $node->value
 >  end
 >  set $node = $node->next
 >end
corrupt node at 0x6120004a2, value=-7      # the loop found it for you
```

The same loop in pdb is just Python, which is the underrated superpower of debugging an interpreted language — the debugger prompt *is* the language, so any expression you can write in the program you can evaluate at the breakpoint:

```python
(Pdb) node = head
(Pdb) while node:
...     if node.checksum != node.expected:
...         print(f"corrupt node {id(node):#x}, value={node.value}")
...     node = node.next
```

Convenience variables also let you remember a pointer across `continue`s — `set $bad = some_ptr` at one breakpoint, then later `print $bad == current_ptr` to test whether you have come back to the same object. That single capability — naming a value now and comparing against it later — is how you confirm a use-after-free interactively: stamp the freed pointer, then watch for it to be handed back out of the allocator.

There is also the convenience function side: gdb exposes the last-printed value as `$` and the one before as `$$`, and the result of value history as `$1`, `$2`, and so on, so you can chain — `print $1->next->next` after a `print head` is `$1`. Small thing, enormous time saver when you are spelunking a deep structure.

## 5. Reverse and replay debugging: running the program backward

This is the feature that feels like cheating, and it is the answer to the hardest debugging question there is: "I see the corrupted value *now*, but what wrote it?" A normal debugger can only go forward. To find the cause of a bad value you have to guess where it might have been written, set a breakpoint, and re-run — and if the bug is non-deterministic, it may not even reproduce. **Reverse debugging** flips this: you stop at the crash and run *backward in time* to the write that caused it.

There are two ways to get it, and the difference matters.

### 5.1 gdb's built-in record (small scope, no extra tools)

gdb can record execution itself with `record`, then step backward with `reverse-step`, `reverse-next`, `reverse-continue`, and `reverse-finish`. It works on any program gdb can run, with no extra tooling — but it records by logging every instruction's effects, so it is *slow* and memory-hungry, suitable for a few thousand instructions, not a whole program run.

```bash
(gdb) break suspicious_region
(gdb) run
(gdb) record                 # start recording from here
(gdb) continue               # ... runs forward until the crash ...
Program received signal SIGSEGV
(gdb) reverse-stepi          # step ONE instruction backward
(gdb) reverse-continue       # run backward to the previous breakpoint/watchpoint
(gdb) reverse-next           # backward over a call
```

Use built-in `record` when the crash is close to a known point and you only need to rewind a short distance. For a whole-program, non-deterministic bug, you want `rr`.

### 5.2 rr: record once, replay deterministically forever

`rr` (from Mozilla) is the production-grade tool. It records an entire run — capturing every source of non-determinism (system call results, signal timing, thread scheduling, the random bits) — into a trace. Then `rr replay` re-executes that *exact* run, deterministically, as many times as you want, forward *and* backward, under a normal gdb. A heisenbug that happens one run in five thousand becomes a recording you can replay on demand and reverse-step through at will.

The mechanism is the heart of why this works, and it is worth making concrete. A program is deterministic *except* at the points where it reads non-determinism from outside: system call return values, the exact interleaving of threads, signal delivery times, reads of the cycle counter or `/dev/urandom`. `rr` records every one of those inputs during the recording run. On replay, instead of *executing* those non-deterministic operations, it *feeds back the recorded values* — so the program follows the identical path, hits the identical bug, every single time. It serializes thread execution onto a single core during recording to make the interleaving reproducible, which is also why recording costs roughly 2–5x in wall-clock time. That cost is the price of a perfectly reproducible bug.

```bash
# 1. Record the (possibly flaky) program. Run it under load / in a loop until
#    the trace you keep is one that crashed.
$ rr record ./service --stress
# ... it crashes ...
$ rr record -n ./flaky_test          # -n records even if it doesn't crash

# 2. Replay that exact run under gdb, deterministically:
$ rr replay
(rr) continue                        # runs forward to the SAME crash, always
Program received signal SIGSEGV
```

### 5.3 The killer combination: hardware watchpoint + reverse-continue

Now the technique that solves use-after-free and memory-corruption bugs in one move. A **watchpoint** is a breakpoint on *data*: gdb tells the CPU's debug registers to trap whenever a specific memory address is written, so you do not have to know *where* in the code the write happens — the hardware catches it. Combine a watchpoint with reverse execution and you get: stop at the crash, watch the corrupted location, run *backward*, and the watchpoint fires at the exact instruction that wrote the bad value. You travel from effect straight to cause.

```bash
$ rr replay
(rr) continue
Program received signal SIGSEGV, Segmentation fault.
0x... in Session::flush () at session.cc:88
(rr) print this
$1 = (Session *) 0x0                 # `this` is the dangling/null pointer
# We want to know who wrote the pointer that is now null. Find the slot:
(rr) up
(rr) print &worker->session          # the address that holds the bad pointer
$2 = (Session **) 0x7ffff7a12340
(rr) watch -l *0x7ffff7a12340        # hardware watchpoint on that slot
(rr) reverse-continue                # run BACKWARD to the last write
Hardware watchpoint: *0x7ffff7a12340
Old value = 0x55556001a000
New value = 0x0
0x... in Worker::reset () at worker.cc:171
171         this->session = nullptr;    # <- THE WRITE. Root cause found.
```

In four commands you went from "segfault on a null `this`" to "`worker.cc:171` set `session = nullptr` and nobody re-initialized it before `tick()` called `flush()`." No guessing where the pointer was nulled. The watchpoint plus reverse-continue *searched time for you*. Figure 4 shows the same idea as a timeline: the store happens, the frame returns and the local dies, the pointer goes dangling, the crash fires thousands of lines later, and `reverse-continue` walks back across all of it to land on the store.

![A timeline tracing a dangling-pointer bug from the store of a stack address, through the frame returning and the local being freed, to the segfault thousands of lines later, then reverse-continue walking back to the originating write](/imgs/blogs/mastering-an-interactive-debugger-4.png)

#### Worked example: rr-replaying a non-deterministic crash, reverse-stepping to the dangling store

A C++ media server crashes with a `SIGSEGV` about one run in 5,000 under load. It never reproduces under the debugger directly — attaching changes the timing and the bug vanishes (the classic heisenbug: observing it perturbs it). So you stop trying to catch it live and record instead:

```bash
# Hammer it under rr until a recording crashes (the trace is saved each run)
$ for i in $(seq 1 8000); do rr record ./mediaserver --bench || break; done
# Run 5,213 segfaulted. That trace is now in ~/.local/share/rr/mediaserver-0/
$ rr replay   # this replays run 5,213 DETERMINISTICALLY, every time
(rr) continue
Program received signal SIGSEGV at frame.cc:233 in Frame::decode()
(rr) print this
$1 = (Frame *) 0x6120000bdeef        # not null — a freed-then-reused address
```

`this` is not null this time — it is a plausible-looking heap address (`0x6120...`), the signature of a **use-after-free**: a `Frame` was freed, its memory handed to another allocation, and we called `decode()` through the now-dangling pointer. Set a watchpoint on the object's vtable slot and reverse-continue:

```bash
(rr) watch -l *(void**)this          # watch the vtable pointer at *this
(rr) reverse-continue
Hardware watchpoint: *(void**)this
Old value = 0x55556... (Frame vtable)
New value = 0x0
0x... in std::__free (this=0x6120000bdeef) at frame_pool.cc:64
64          ::operator delete(p);    # <- the Frame was freed HERE
(rr) reverse-continue                # keep going back: who called free?
0x... in Decoder::recycle () at decoder.cc:118
118         pool.release(current_frame_);   # released while still referenced
```

The recording is run 5,213 forever; you can replay it as many times as you need, reverse-step as finely as you like, and the bug is identical every time. The root cause: `Decoder::recycle()` released `current_frame_` back to the pool while a decode of that same frame was still in flight on another path — a use-after-free that depends on a one-in-5,000 interleaving. You found it not by reproducing it 5,000 more times, but by recording it *once* and running time backward. The before→after is stark: before, an unreproducible crash that vanished under observation; after, a deterministic recording pinned to `decoder.cc:118` that you can show a colleague on demand.

This is the deepest expression of the **bisect** step in the debugging loop — binary-searching not through commits or stack frames but through *execution time itself*, from a symptom backward to its cause.

### 5.4 Why a hardware watchpoint is fast and what its limits are

It is worth being precise about *why* the watchpoint move is cheap, because the alternative — a software watchpoint — is so expensive it changes what is practical. A **hardware watchpoint** programs the CPU's debug registers (on x86, the `DR0`–`DR3` registers plus `DR7`) with an address and a condition; the processor itself raises a debug exception the instant any instruction writes that address. There is *zero* per-instruction overhead — the program runs at full native speed and only stops when the watched location actually changes. The catch is that there are only a handful of these registers (four on x86), and each can watch only a small, aligned region (1, 2, 4, or 8 bytes). So you can watch a pointer, a counter, a flag — but not a whole 4 KB struct.

A **software watchpoint** is the fallback when the hardware cannot do it (too many watchpoints, too large a region, or a platform without debug registers). gdb implements it by single-stepping the program and checking the value after *every instruction*. That is correct but catastrophically slow — often 100x or worse — which is why `watch` on a large object can make a program crawl. The lesson: watch the *smallest* thing that pins the bug (the pointer, not the struct it points into), and you stay on the fast hardware path. When you must watch something big, narrow it first with a conditional breakpoint to get close, *then* set the watchpoint for the final approach.

Reverse execution composes with this beautifully precisely because the watchpoint is a hardware trap: under `rr replay` the watchpoint fires in reverse with the same near-zero overhead, so "run backward to the last write of this 8-byte pointer slot" is both *correct* and *fast*. That combination — deterministic replay plus a hardware watchpoint plus reverse-continue — is the single most powerful move in the entire debugging toolkit, and it is the one the fewest engineers have ever used. If you take one technique away from this post, take that one.

## 6. Remote and cross debugging

Your bug is on a machine you cannot run a desktop debugger on: an embedded ARM board, a tiny container, a locked-down production host, a customer's appliance. The answer is to split the debugger in two. `gdbserver` runs on the *target*, controlling the inferior process; full `gdb` runs on *your* machine, where you have the source and symbols; they talk over a wire (TCP or serial). Figure 6 shows the split.

![A layered diagram showing host gdb on a laptop with a source-path substitution map, connecting over a TCP or serial link to gdbserver on the target, which controls the inferior process on an embedded board, resulting in breakpoints hitting remotely](/imgs/blogs/mastering-an-interactive-debugger-6.png)

```bash
# On the TARGET (embedded board, container, remote host):
target$ gdbserver :2345 ./buggy            # launch under gdbserver, port 2345
# or attach to a process already running there:
target$ gdbserver --attach :2345 8421

# On the HOST (your laptop, with source + symbols):
host$ gdb ./buggy                          # the SAME binary, with debug info
(gdb) target remote 192.168.1.50:2345      # connect across the network
(gdb) break main
(gdb) continue                             # breakpoints fire on the target
```

The subtlety that bites everyone is **source path mapping**. The binary was *built* in a CI container at `/build/src/...` but your source lives at `/home/you/project/...`. gdb has the line numbers but cannot find the files, so it shows addresses instead of source. Two settings fix it:

```bash
(gdb) set substitute-path /build/src /home/you/project   # remap the prefix
(gdb) set sysroot /path/to/target/rootfs                 # for cross libs
(gdb) directory /home/you/project/src                    # extra source dirs
```

`delve` does remote the same way (`dlv --headless --listen=:2345 attach PID` on the target, `dlv connect 192.168.1.50:2345` on the host), and for cross-architecture work you point gdb at the cross toolchain's `gdb` build (e.g. `arm-none-eabi-gdb`) so it understands the target's instruction set. Node has the analogous `node --inspect=0.0.0.0:9229` plus a Chrome DevTools client connecting across the wire. The pattern is universal: thin server on the target, full client where the source lives, a path map to bridge them.

## 7. Multi-thread control: stop guessing which thread

Concurrency bugs are where the debugger's *control* features earn their keep, because the default behavior — stop *all* threads when any breakpoint hits — is sometimes exactly wrong, and reading a multi-thread state requires knowing how to drive each thread independently.

### 7.1 The mechanism: why a deadlock is a cycle

First, *why* the deadlock from Section 2 is a deadlock, made rigorous, because reading it off the stacks depends on understanding it. Model the locks and threads as a directed graph. Draw an edge from a thread to a lock it is *waiting for*, and an edge from a lock to the thread that *holds* it. A deadlock exists **if and only if** this wait-for graph contains a cycle. In our case: thread 18 waits-for `accountMu` → `accountMu` held-by thread 24 → thread 24 waits-for `ledgerMu` → `ledgerMu` held-by thread 18 → back to the start. A cycle of length two. Because each thread will only release its held lock *after* acquiring the one it waits for, and it can never acquire it (the holder is itself blocked in the cycle), the cycle is permanent. No timeout, no progress, forever. Figure 3 draws exactly this cycle.

![A graph showing thread 7 holding lock A and waiting on lock B while thread 12 holds lock B and waits on lock A, the two wait edges meeting at a detected cycle that resolves to a fixed global lock order](/imgs/blogs/mastering-an-interactive-debugger-3.png)

That is the mechanism, and it is *why* the move in Section 2 works: dump every thread's stack, note for each blocked thread which lock it is parked acquiring and which it already holds (the held locks are visible in the frames above the blocking call, or via `info threads` and the mutex internals), and look for a cycle. Find the cycle, and the fix writes itself: impose a **global lock ordering** — every code path that needs both locks must acquire them in the same order (say, always `accountMu` before `ledgerMu`). With a consistent order, no cycle can form, because the wait-for graph becomes a DAG by construction. (ThreadSanitizer and helgrind can find these *before* they deadlock by watching lock-acquisition order across runs; that is the prevention angle, covered in the planned `deadlocks-livelocks-and-starvation` sibling.)

### 7.2 The method: scheduler-locking and thread-specific breakpoints

When you continue or step in a multi-threaded program, gdb's default is to let *all* threads run. That makes stepping through one thread's logic maddening — other threads race ahead and the state changes under you. `set scheduler-locking on` freezes every thread except the one you are stepping, so you can walk one thread's logic in isolation:

```bash
(gdb) info threads                  # see all threads; * marks the current one
(gdb) thread 7                      # switch to thread 7
(gdb) set scheduler-locking on      # ONLY thread 7 runs when you step/continue
(gdb) step                          # step thread 7 alone; others stay frozen
(gdb) set scheduler-locking step    # safer middle ground: lock during stepping,
                                    # let all run on `continue`
(gdb) set scheduler-locking off     # back to default (all threads run)
```

Use `scheduler-locking on` with care: if the thread you froze everything for is waiting on a lock held by a thread you just froze, *you* have deadlocked the debugging session. `scheduler-locking step` is the pragmatic default — single-stepping is single-threaded (so it is comprehensible), but `continue` lets everyone run (so you do not self-deadlock).

A **thread-specific breakpoint** fires only when a *named* thread reaches it — invaluable when twenty threads run the same function but only one misbehaves:

```bash
(gdb) break worker.cc:204 thread 7        # only thread 7 stops here
(gdb) break process if pthread_self() == target_tid   # by predicate
```

delve has the equivalents (`goroutine N`, `goroutines`, and breakpoints scoped via conditions on `runtime.curg`), and lldb mirrors gdb (`thread select N`, `settings set target.process.thread.step-avoid-...`, breakpoints with `-T thread-name`).

Non-stop mode is the advanced cousin: `set non-stop on` lets you stop *one* thread while the others keep running — essential when stopping all threads would break the system (you cannot freeze the heartbeat thread of a clustered service, or its peers evict it). In non-stop mode you debug a misbehaving worker while the healthy threads keep serving.

![A matrix mapping each symptom class of hung process, hard crash, and non-deterministic flake to its best debugger entry door, the specific tool and flag, and the cost of using it](/imgs/blogs/mastering-an-interactive-debugger-2.png)

The matrix in Figure 2 collects the whole decision: match the *symptom* to the entry door. A hang wants attach + all-thread-stacks; a hard crash wants a core dump; a non-deterministic flake wants record-replay. Each carries a cost — attach pauses prod, cores need to be enabled, recording slows the run — and the right choice is the one whose cost you can afford for the evidence you need.

## 8. A note on the order of operations: don't waste the freeze

A subtle but important discipline ties the live techniques together: **when you stop a process, you are spending a budget.** Attaching freezes prod; recording slows it; a core dump captures one instant. So the order of operations matters. Decide *before* you stop what you need to capture, capture it fast, and let go.

The pattern for a live production hang:

1. **First, a read-only snapshot** that costs almost nothing: `py-spy dump`, `jstack`, or `kill -USR1` into a `faulthandler` handler. This often answers "where is it stuck?" without a real freeze.
2. **If you need state, `gcore` it** — dump a core in a few hundred milliseconds, then *let the process run* and study the core offline. This is the single best habit in production debugging: you get the full forensic snapshot without holding the service hostage.
3. **Only attach interactively as a last resort**, and when you do, run a script (`thread apply all bt full` and a `generate-core-file`) and detach immediately rather than poking around live.

This is the same evidence-first instinct as the broader [scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging): capture before you theorize, and never destroy the evidence (by restarting) before you have it. Figure 8 makes the trade-off concrete — "restart and hope" throws away the one state that holds the bug and leaves you praying for a 1-in-5,000 repro; "capture then study" freezes the evidence and turns hours of MTTR into minutes.

![A before-and-after comparison showing that killing a wedged process destroys the state and leaves a tiny chance of reproduction, while attaching or recording first preserves the evidence and cuts mean time to root cause from hours to minutes](/imgs/blogs/mastering-an-interactive-debugger-8.png)

## 9. War story: the bug that only existed between two instructions

Consider a real category of failure made famous by the hardest concurrency bugs ever shipped: a value that is correct on the instruction before and the instruction after, but wrong for the single instruction in between — a torn read or a check-then-act race (TOCTOU, time-of-check to time-of-use). The Therac-25 radiation therapy machine, in the 1980s, killed patients because of exactly this shape of bug: a race condition where a fast operator could change the treatment setup in the narrow window between the software checking a value and acting on it, and the safety interlock that should have caught it had been moved from hardware to software where the race could defeat it. The bug was real, intermittent, and for a long time un-reproducible — it only manifested when an experienced operator typed faster than the developers had ever tested. It is the canonical lesson that *intermittent does not mean rare-and-harmless; it means you have not yet found the interleaving.*

Now imagine debugging Therac-25's race today. You could not catch it with a breakpoint — stopping the program changes the timing and the window closes (the heisenbug problem in its purest form). But you *could* record a session under `rr` where an operator reproduced the fast-edit sequence, then replay it deterministically and reverse-step across the exact two-instruction window to watch the stale value get used. The interleaving that took years to understand in the 1980s would be a deterministic recording you replay on a laptop. That is not hindsight smugness — it is the precise reason record-replay exists: to make the un-reproducible reproducible, so the bug between two instructions becomes a thing you can stop on, inspect, and run backward from.

A second, more modern shape: the Knight Capital trading loss of 2012, where a deploy left old code (a repurposed feature flag) active on one of eight servers, and that one server began firing erroneous orders at machine speed — about \$440 million lost in 45 minutes. The relevant debugging lesson is *not* that a debugger would have caught it (it was a deployment and dead-code problem), but that when the symptom is "one host out of eight behaves differently," the move is to attach to the *misbehaving host's process specifically* and compare its live state — its loaded code paths, its feature flags in memory — against a healthy peer. `print` the flag's value in the running process on the bad host; you would have seen the stale path armed. The discipline is the same one from the start of this post: go to where the bug is alive, look at the actual running state, and do not theorize about what *should* be loaded when you can read what *is*.

The thread connecting both stories — and the reason this whole post matters — is that the hardest production bugs are the ones that vanish the instant you do the obvious thing. Restart the wedged pod and the deadlock is gone. Stop the racing program with a breakpoint and the window closes. Redeploy the bad host and the stale flag clears. Each of those obvious reflexes destroys the evidence. The techniques in this post are, at heart, a single discipline expressed five ways: *get to the live or recorded state before you destroy it, look at what is actually there, and let the tool — not your intuition — search the space.* Attaching preserves the deadlock. A core dump preserves the crash. A `gcore` snapshot preserves a hang while letting prod recover. An `rr` recording preserves a one-in-5,000 race forever. And a watchpoint plus reverse-continue lets the hardware, not your guesses, find the write that started it all. The engineer who has internalized these does not panic at 3am; they capture, then study.

## 10. How to reach for this (and when not to)

Every technique here has a cost, and a principal engineer's value is partly in knowing when *not* to reach for the heavy tool. The decision, distilled, is in Figure 7: the symptom's shape picks the feature.

![A decision tree branching on whether the process is live, gone, or firing a breakpoint too often, leading respectively to attach with all thread stacks, a core dump, record-replay, or a dprintf and conditional breakpoint](/imgs/blogs/mastering-an-interactive-debugger-7.png)

**Reach for attaching when** a process is alive but wrong and restarting would destroy the evidence — a hang, a leak, a stuck pool. **But do not attach interactively to a latency-sensitive production process** and start poking; the freeze fails health checks and can cascade. Use `py-spy dump` / `gcore` to capture without a real stop, and study offline.

**Reach for a core dump when** the process crashes, especially intermittently — it converts "segfault, no idea" into "null `this` at this line." **But cores can contain secrets** (keys, customer data live in that memory), so handle them like production data: restricted access, scrubbed, deleted after. Do not casually email a core dump of the payments service.

**Reach for scripting / dprintf when** you need logging from a binary you cannot rebuild, or a breakpoint that must fire on the 3.8-millionth iteration. **But if one well-placed log line and a redeploy answers the question, do that instead** — a `dprintf` you have to remember to set up every session is worse than a permanent log line your teammates also benefit from. The debugger script is for when you cannot or should not change the code.

**Reach for rr / reverse debugging when** the bug is non-deterministic or you are chasing a corruption back to its write — it is unmatched for use-after-free, heisenbugs, and "what wrote this?" **But rr has costs and limits**: 2–5x recording slowdown, it is Linux-x86/ARM only, it needs specific CPU performance-counter access (tricky in some VMs and containers), and it does not love programs that depend on hardware you cannot record. Do not record your entire integration suite by default; record the *one* flaky test, under a loop, until you catch a crashing trace.

**Reach for remote debugging when** the bug is on a box you cannot run a full debugger on. **But get the source-path mapping right first** — a remote session showing hex addresses instead of source is worse than useless. And **never leave a `gdbserver` listening on an open port** in production; it is a remote code execution primitive by design.

And the meta-rule from the intro to this whole craft: **do not reach for a debugger at all when a hypothesis and a one-line experiment would settle it faster.** The debugger is for when you genuinely do not know where the bug is. When you have a strong hypothesis, the cheapest confirming test — a log line, an assertion, a unit test that pins the boundary — often beats a debugger session. The scientific method comes first; the debugger is how you *observe* when observation is the hard part.

## How to reach for this (and when not to): the short version

| Symptom | Entry door | Command | Do not |
| --- | --- | --- | --- |
| Hung / wedged, alive | Attach + all stacks | `thread apply all bt`, `dlv goroutines -t`, `py-spy dump` | Restart before capturing; freeze a latency-critical proc interactively |
| Crashed / segfault | Post-mortem core | `gdb prog core`, `coredumpctl gdb`, `dlv core` | Forget `ulimit -c`; leak secrets in the core |
| Fires too often / huge loop | Conditional bp + dprintf | `break ... if cond`, `dprintf`, `ignore N M` | Use it where a permanent log line is better |
| Non-deterministic flake | Record-replay | `rr record` then `rr replay` + `reverse-continue` | Record everything by default; assume it works in every VM |
| Multi-thread / deadlock | Thread control | `set scheduler-locking`, thread-specific bp | Freeze the thread holding the lock you need |
| Remote / embedded | gdbserver split | `gdbserver :2345` + `target remote` | Skip `set substitute-path`; leave the port open in prod |

## Key takeaways

- **You almost never need to restart.** Attach to a live process, dump a core from it without killing it (`gcore`), or replay a recording. Restarting destroys the one state that holds the bug.
- **For any hang, the first command is "dump every thread's stack."** `thread apply all bt`, `dlv goroutines -t`, `py-spy dump`, `jstack`, or a `faulthandler` `SIGUSR1` handler. A deadlock reads off the stacks as a cycle of who-holds-and-who-waits.
- **A core dump turns "segfault, no idea" into "null `this` at this line."** Enable cores (`ulimit -c unlimited` / `coredumpctl`), or generate one on demand with `gcore` / `GOTRACEBACK=crash` / `faulthandler`, then open it with `gdb prog core` / `dlv core`.
- **The debugger is programmable.** Breakpoint command lists, `dprintf`, conditional breakpoints, the gdb Python API, pretty-printers, and `.gdbinit` turn a repetitive investigation into one unattended run — even a repeat-until-fail loop that captures a core only on the failing run.
- **Reverse debugging answers "what wrote this?"** Record with `rr`, replay deterministically, set a hardware watchpoint on the corrupted location, and `reverse-continue` to the exact write — the deepest form of binary-searching backward through time.
- **Respect the cost of stopping.** Attaching freezes the target; recording slows it; capture read-only first (`py-spy dump`), `gcore` second, attach interactively last.
- **Match the tool to the symptom, not to habit.** Alive and wrong → attach. Dead → core. Flaky → replay. Too-frequent → conditional breakpoint. On a box you cannot reach → gdbserver.
- **The scientific method comes first.** When a one-line hypothesis test would settle it, the debugger is overkill. Reach for these power features when observation itself is the hard part.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe → reproduce → hypothesize → bisect → fix → prevent loop this post lives inside.
- [The debugger is a microscope: use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) — the Track-A intro to breakpoints, watchpoints, and stepping that this post builds on.
- [Reproduce it first, or you are not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — the repeat-until-fail discipline behind the scripted-debugger and rr-record loops here.
- `reading-a-core-dump-post-mortem-analysis` (planned Track-E sibling) — the deep companion on reading core dumps in anger; this post covers how to *generate* and *open* one, that one covers how to *read* one.
- `deadlocks-livelocks-and-starvation` (planned sibling) — the concurrency theory behind the wait-for-graph cycle and the global-lock-ordering fix sketched in Section 7.
- [Debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale) — attaching, namespaces, and live-state inspection across a distributed fleet.
- The **GDB manual** (especially the chapters on Process Record/Replay, Python API, and Remote Debugging), the **`rr` project documentation** (`rr-project.org`), the **Delve documentation**, and David Agans' *Debugging: The 9 Indispensable Rules* and Andreas Zeller's *Why Programs Fail* for the discipline underneath the tools.
