---
title: "Seeing What a Process Really Does: Syscall Tracing From strace to bpftrace"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "When the bug lives at the boundary between your code and the operating system, syscall tracing shows you the truth your source hides — the wrong path it actually opened, the host it really connected to, the syscall it is frozen inside — using strace, ltrace, dtruss, and bpftrace with real flags and annotated output."
tags:
  [
    "debugging",
    "software-engineering",
    "strace",
    "ltrace",
    "bpftrace",
    "ebpf",
    "syscalls",
    "linux",
    "ptrace",
    "observability",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-1.png"
---

The config file is right there. You can `cat` it in the terminal. It has the database URL, the timeout, the feature flag your whole afternoon depends on. The service starts, prints `loaded configuration`, and then behaves as if the file does not exist — defaults everywhere, the flag off, the timeout at some value you never set. You read the config loader. It looks correct. You add a print statement; it prints the path you expect. You read it again. You start to doubt your own eyes. Two hours in, you are no longer debugging the program; you are debugging your belief about the program.

Here is the trap, and it is the same trap in a hundred different disguises: your source code describes what the program *intends* to do, and you have been reading intent. But intent is not behavior. Somewhere between the line of code you are staring at and the bytes on disk, a system call happened — `openat("/etc/app.d/app.conf", O_RDONLY)` returned `ENOENT`, the program shrugged, fell back to defaults, and printed a cheerful success message anyway. The kernel saw the truth. Your source hid it from you. And there is a tool that makes the kernel tell you, in one line, exactly which path the program tried and exactly what the answer was.

This post is about that boundary — the thin, hard line between your process and the operating system — and about the family of tools that let you watch every crossing of it. Every file your program opens, every socket it connects, every byte it reads or writes, every page it maps, every child it forks, every lock it sleeps on: all of it goes through a **system call** (a "syscall"), a controlled entry into the kernel. The kernel is the one component that cannot lie to you, because it is the one doing the work. Figure 1 shows the geometry: your code's intent funnels down through the C library, across the syscall boundary, into the kernel — and a tracer sits at that boundary recording what really crossed.

![A flow diagram showing application code passing through the libc wrapper and across the narrow syscall boundary into the kernel, with strace and bpftrace observers attached at the boundary, all arriving at the ground truth of what the program actually did](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-1.png)

By the end you will be able to: take a program that "isn't reading the config" and find the wrong path in under two minutes; watch a service make a `connect()` to a host nobody expected, and catch a silent `EACCES` permission denial that swallowed an error; attach to a frozen process with zero CPU and read the exact syscall it is parked inside; run a one-line summary that finds the program making two million `stat()` calls; and do all of this in production with `bpftrace` instead of `strace` when the overhead matters. This is the **observe → reproduce → hypothesize → bisect → fix → prevent** loop from [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging), aimed squarely at the layer where your code meets the machine. Syscall tracing is the purest *observe* tool we have: it does not ask the program what it did; it watches it do it.

## 1. The mechanism: why a syscall is the one place the truth lives

Start with the wall that makes this whole technique possible. Your program runs in **user mode**, a restricted CPU privilege level where it cannot touch hardware, cannot read another process's memory, cannot write to disk directly, cannot send a packet on the network. None of those things are ordinary function calls. They are powers the kernel reserves for itself, running in **kernel mode**. The only way a user-mode program can reach across that wall and ask for one of those powers is to execute a special CPU instruction — on x86-64 Linux it is the `syscall` instruction — that traps into the kernel at a fixed, controlled entry point.

So when your Python code writes `open("/etc/app.conf")`, an enormous amount happens, but it funnels to one moment. Python calls into libc (or makes the call directly), libc loads a number into a register (`SYS_openat`, which is `257` on x86-64 Linux), loads the path pointer and the flags into other registers, and executes `syscall`. The CPU switches privilege level, jumps to the kernel's syscall dispatch table, the kernel does the actual work of walking the filesystem and checking permissions, and then returns a value in a register — a file descriptor on success, or a negative error code that libc turns into `-1` and an `errno`. That single instruction is the choke point. **Every interaction your program has with the outside world has to pass through it.** There is no back door.

That choke point is what makes syscall tracing so devastatingly effective as a debugging tool, and it is worth being precise about why. A print statement shows you what the programmer *decided to print*. A log line shows you what the programmer *thought was worth logging*. A stack trace shows you where execution *was*. None of them are guaranteed to reflect reality, because all of them are written by the same fallible human whose mental model is the thing under suspicion. The syscall stream is different in kind: it is generated by the kernel, mechanically, as a side effect of the program actually doing the work. The program cannot do file I/O without the kernel seeing the exact path. It cannot make a network connection without the kernel seeing the exact address. The tracer reads it straight from the source of truth.

There is a second, subtler reason this layer is special. The syscall interface is a *narrow* and *stable* contract. The Linux kernel has on the order of 350 syscalls, and they almost never change their meaning, because the kernel's promise to never break userspace lives precisely here. That stability means the same handful of trace filters — `openat`, `connect`, `read`, `futex`, `stat`, `mmap` — answer the same handful of debugging questions across every language and every framework on the machine. A Go binary, a Python script, a JVM service, a Rust daemon: under the trace they all dissolve into the same vocabulary of syscalls. You stop debugging "the framework" and start debugging "the process," and the process is simple at this layer in a way it never is at the source layer.

The one idea that the whole rest of the post hangs on is this: **the gap between what your code says and what your program does is exactly the gap a syscall trace measures.** Most boundary bugs — wrong path, wrong host, missing permission, silent fallback, a hang on a lock — are gaps of precisely this kind. They survive code review because the code *reads* correct. They survive print debugging because the print reflects the same wrong belief. The trace does not share the belief, so it does not share the blindness.

### How the crossing physically works: registers and the ABI

It helps to be concrete about the *physical* mechanism of a syscall, because once you have seen it you will never again think of `open()` as "just a function call," and you will understand why a tracer can read every argument with perfect fidelity. On x86-64 Linux there is a fixed contract — the **syscall ABI** — for how arguments are passed across the boundary. The syscall *number* goes in the `rax` register. The arguments go, in order, in `rdi`, `rsi`, `rdx`, `r10`, `r8`, `r9`. Then the CPU executes the `syscall` instruction, which jumps to a fixed kernel entry point; when the kernel is done, it places the return value back in `rax`, and the instruction after `syscall` resumes in user mode.

So `openat(AT_FDCWD, "/etc/app.conf", O_RDONLY)` compiles, at the metal, to: put `257` (the number for `openat`) in `rax`; put `AT_FDCWD` (the value `-100`) in `rdi`; put the *address of the path string* in `rsi`; put `O_RDONLY` (the value `0`) in `rdx`; execute `syscall`; read the result fd out of `rax`. That is the entire crossing. There is nothing else — no hidden channel, no second path. This is precisely why a tracer is so reliable: at the moment the kernel stops the process on syscall-entry, the tracer reads `rax` to learn *which* syscall and reads `rdi`/`rsi`/`rdx`/… to learn the arguments, and because the second argument is a *pointer* to the path string, strace follows that pointer into the traced process's memory and reads the actual bytes — which is how it can print `"/etc/app.conf"` and not just an address. On syscall-exit the kernel stops the process again and the tracer reads `rax` for the return value. The decoded line you see — `openat(AT_FDCWD, "/etc/app.conf", O_RDONLY) = 3` — is strace translating those raw register values back into human-readable names. There is no possibility of the program "hiding" an argument, because the kernel must be handed the real value in a real register to do the work at all.

This register-level reality also explains a class of confusing trace output. When you see `connect(7, {sa_family=AF_INET, sin_port=htons(5432), sin_addr=inet_addr("10.4.2.9")}, 16)`, strace is dereferencing the pointer in `rsi` to a `sockaddr_in` struct in the process's memory and decoding its fields. The `16` is the third argument (`rdx`), the length of that struct. strace knows the *shape* of every syscall's arguments — which are scalars, which are pointers to structs, which are pointers to buffers — and decodes each appropriately. That knowledge is what turns an unreadable register dump into the annotated line you actually read. It is also why `strace -s 256` matters: by default strace only copies the first 32 bytes of a string or buffer out of the traced process's memory; for inspecting the *contents* of a `write` (what exactly did the program send on this socket?) you raise that limit so you see the full payload, not a truncated head.

## 2. How a tracer actually attaches: ptrace and the cost it imposes

Before we run anything, you should know how `strace` works, because its mechanism is also its main limitation, and you will make better decisions if you understand the cost you are paying. `strace` is built on a single Linux system call named `ptrace` — the same primitive that `gdb` uses to attach to a process. (If you have read the companion piece on [mastering an interactive debugger](/blog/software-development/debugging/mastering-an-interactive-debugger), this will be familiar; `ptrace` is the shared engine under both.)

When you run `strace ./app`, strace forks, the child calls `ptrace(PTRACE_TRACEME)` to mark itself as traced, and then execs your program. From then on the kernel does strace a favor: every time the traced process is about to enter a syscall, the kernel stops it and hands control to strace; strace reads the registers to see which syscall it is and what the arguments are, then lets it proceed; when the syscall returns, the kernel stops the process *again* so strace can read the return value. Two stops per syscall. Each stop is a context switch out of your process, into the kernel, into strace, and back.

That is the heart of the overhead problem, drawn in Figure 8 later in this post. For a process that makes a few hundred syscalls a second — most interactive programs, most request handlers — the cost is invisible. For a process in a hot loop doing half a million syscalls a second, those two context switches per call multiply out to a process running **10 to 100 times slower under strace** than without it. This is not a rounding error. It can be the difference between a program that completes and one that appears to hang. It is also why you must never reflexively `strace` a busy production process and walk away; you can turn a slow service into a dead one.

The practical consequence is a rule you will use constantly: **always filter.** Unfiltered `strace` traces every syscall, pays the stop cost on every one, and buries the signal in a flood of `mmap`, `brk`, `rt_sigprocmask`, and `futex` noise. Filtered strace — `-e trace=openat`, `-e trace=network` — tells the kernel to stop the process *only* for the syscalls you care about, which both makes the output readable and slashes the overhead because the uninteresting syscalls run at full speed. We will filter in nearly every command below.

Here is the canonical first invocation, with the flags you will reach for most:

```bash
# -f      follow forks/clones (children too) — essential for any real service
# -e trace=openat   only show file-open syscalls
# -o out  write to a file instead of mixing with the program's own stderr
strace -f -e trace=openat -o trace.txt ./app

# Timing flavors, added when "what" is known and "how long / when" is the question:
strace -T  -e trace=network ./app   # -T appends the elapsed time of EACH syscall
strace -tt -e trace=network ./app   # -tt prefixes a wall-clock timestamp (microsecond)
strace -c  ./app                    # -c prints a SUMMARY (count + time per syscall), no per-call lines
```

A note on `ltrace`, strace's sibling. Where strace intercepts *syscalls* (the kernel boundary), `ltrace` intercepts *library calls* (the boundary one ring up — `malloc`, `fopen`, `getaddrinfo`, your own dynamically linked functions). It works by a related trick: it overwrites the Procedure Linkage Table entries with breakpoints so that each call into a shared library traps. It is invaluable when the bug is in the logic *above* the syscall — for example, the program computes the wrong path *before* it ever calls `openat`, so the syscall trace is correct but the input to it is wrong, and you need to see the `sprintf`/`getenv`/`fopen` that produced it. The trade-off: `ltrace` is often even slower and flakier than strace, and on modern toolchains with full RELRO it sometimes cannot hook the PLT at all. Reach for it deliberately, not by default.

## 3. The most common bug there is: the file it actually opened

Let me make the opening scenario concrete, because the "config isn't being read" bug is, in my experience, the single most common thing syscall tracing fixes, and it fixes it faster than any other technique by a wide margin. Figure 2 shows the layered picture of where the different tracers attach for exactly this kind of problem.

![A layered stack diagram showing application code, the library call layer hooked by ltrace, the syscall layer hooked by strace, the kernel function layer hooked by bpftrace kprobes, and the hardware and filesystem at the bottom](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-2.png)

The setup: a service is supposed to read `/etc/app/app.conf`. You have created that file. The service starts, logs `loaded configuration`, and runs with defaults. You have read the config-loading code three times. It constructs a path, opens it, parses it. Everything looks right. The `loaded configuration` log even prints — which is the cruelest part, because it convinces you the load succeeded.

Stop guessing. Watch the opens:

```bash
strace -f -e trace=openat -e signal=none ./app 2>&1 | grep -i 'app\|conf'
```

The `-e trace=openat` filter shows only file-open syscalls; the `grep` narrows to the lines mentioning your config. Within a fraction of a second, you see something like this in the output:

```bash
openat(AT_FDCWD, "/etc/app.d/app.conf", O_RDONLY) = -1 ENOENT (No such file or directory)
openat(AT_FDCWD, "/usr/share/app/defaults.conf", O_RDONLY) = 3
```

There it is, in two lines, the entire mystery solved. The program tried to open `/etc/app.d/app.conf` — note the `.d` — got `ENOENT` ("No such file or directory"), and then fell back to opening the *defaults* file (which succeeded, returning file descriptor `3`). Your file is at `/etc/app/app.conf`. The program looks in `/etc/app.d/`. Off by a single `.d`, a directory that probably exists for some unrelated drop-in mechanism. The "loaded configuration" log was printed after the *defaults* loaded, not your file. You never had a chance reading the source, because the path was assembled from a base directory constant, an environment variable, and a join, and the bug was in the constant, which *looked* fine in isolation.

The fix takes two minutes: move the file, or set the environment variable the loader actually reads, or fix the constant. The point is not the fix; the point is that the trace *named the exact wrong path* and *named the exact error*, and it did so without you understanding the config loader at all. Figure 4 contrasts the two approaches directly — the hour of reading versus the two-minute trace.

![A two-column before and after diagram contrasting an hour of guessing from the source against a single openat trace that named the wrong directory and the missing file error, leading to a two-minute fix](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-4.png)

This same trace solves a whole family of cousins. The library it can't find (`openat` walking the entire `LD_LIBRARY_PATH` and `RUNPATH`, every one returning `ENOENT`, until the last). The certificate it reads from the wrong bundle. The `.env` file it loads from the working directory instead of the project root because someone launched it from `/`. The unit-test fixture it can't locate because the test runner changed the current directory. In every case, the trace prints the exact path and the exact result, and the bug becomes obvious the instant you see it written down.

#### Worked example: the config that "isn't being read"

A teammate spent the better part of a morning on this exact bug. The application was a Go service; its config path came from `filepath.Join(configDir, "app.conf")`, and `configDir` defaulted to a build-time constant set by an `ldflags` injection. The injection had a stray `.d` from a copy-paste off a different service's Makefile. Reading the Go source, everything was clean — `configDir` was a variable, you could not see its value, and the literal default in the source code was actually correct; the wrong value came from the linker flag, which lives in the build system, not the source.

The trace cut through all of it. One command — `strace -f -e trace=openat ./service 2>&1 | grep app.conf` — printed `openat(AT_FDCWD, "/etc/app.d/app.conf", O_RDONLY) = -1 ENOENT`. Total time from "I'm stuck" to root cause: under sixty seconds. The fix was a one-character change in the Makefile. The lesson the teammate took away, and the reason I am telling this story, is that **no amount of reading the source would have found it, because the bug was not in the source** — it was in a value injected at build time that the source could not show. The trace does not read the source. It reads reality.

## 4. The network: the wrong host, the silent denial, the hung lookup

The second great category of boundary bug lives in the network syscalls, and it is where tracing pays off most dramatically, because network problems are notorious for failing *silently* or *slowly* rather than loudly. The filter is `strace -e trace=network`, which shows `socket`, `connect`, `bind`, `sendto`, `recvfrom`, `getsockopt`, and friends. Three classic failures fall out of it.

**The unexpected connection.** Your service is supposed to talk to `db.internal:5432`. It is timing out. You add the network filter:

```bash
strace -f -e trace=network -e trace=connect ./app 2>&1 | grep -i connect
```

and you see:

```bash
connect(7, {sa_family=AF_INET, sin_port=htons(5432), sin_addr=inet_addr("10.4.2.9")}, 16) = -1 EINPROGRESS (Operation now in progress)
```

The connection is going to `10.4.2.9`, but `db.internal` should resolve to `10.4.2.7`. Someone's stale entry — in DNS, in `/etc/hosts`, in a service-discovery cache, in a hardcoded fallback — is sending you to the wrong box, which happens to have nothing listening on `5432`, so you hang on `connect` until the timeout. The trace printed the IP. You did not have to guess which of four config layers held the wrong value; you saw the value the kernel was actually handed.

**The silent permission denial.** This one is brutal because the program often swallows it. You see:

```bash
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = 5
connect(5, {sa_family=AF_INET, sin_port=htons(443), sin_addr=inet_addr("142.250.1.1")}, 16) = -1 EACCES (Permission denied)
```

`EACCES` on a `connect` to port 443 is the fingerprint of an egress firewall, an SELinux policy, a seccomp filter, or a container network policy blocking outbound traffic. The application code probably caught the error, logged a generic "connection failed," and moved on, so the *real* cause — a deliberate denial, not a network outage — was invisible. The distinction matters enormously for the fix: a network outage you wait out; a policy denial you fix in the policy. The trace tells you which by printing `EACCES` instead of `ECONNREFUSED` or `ETIMEDOUT`. Three different errors, three completely different root causes, all indistinguishable from inside the application's catch block, all named precisely in the trace.

**The DNS lookup that hangs.** When a service "hangs on startup," resolution is a prime suspect, and it hides from the network filter because name resolution is not a single syscall — `getaddrinfo` underneath does a `connect` to the resolver, `sendto` the query, and a `recvfrom`/`poll` for the answer, possibly over several name servers with timeouts. To see it you widen the filter and add timestamps:

```bash
strace -f -tt -T -e trace=network,poll ./app 2>&1 | tail -40
```

`-tt` stamps each line with a microsecond wall-clock time and `-T` appends how long the syscall took. A hung lookup shows up unmistakably as a `connect` to the resolver on port 53 followed by a `poll` that sits for five full seconds before timing out and retrying the next nameserver:

```bash
15:32:01.004 connect(8, {sa_family=AF_INET, sin_port=htons(53), sin_addr=inet_addr("10.0.0.2")}, 16) = 0 <0.000040>
15:32:01.004 sendto(8, "...", 41, MSG_NOSIGNAL, NULL, 0) = 41 <0.000061>
15:32:06.005 poll([{fd=8, events=POLLIN}], 1, 5000) = 0 (Timeout) <5.000812>
```

That `<5.000812>` is the smoking gun: a five-second `poll` that timed out. The first nameserver (`10.0.0.2`) is unreachable, so every lookup eats a five-second timeout before failing over. Multiply by every name your service resolves at startup and you have a thirty-second "hang" that is really six failed DNS timeouts in a row. Again — the trace did not just say "it's slow"; it said *which* syscall was slow, *how* slow (`<5.000812>`), and *to whom* (`10.0.0.2`).

The decision tree in Figure 5 ties these symptoms to filters: a missing file routes to `openat`, a network problem to the network filter, a freeze to attaching to the live PID, a churn to the summary.

![A decision tree routing from a misbehaving process down through four symptom branches for missing file, wrong host or hang, frozen process, and high CPU churn, each leading to a specific strace or bpftrace filter](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-5.png)

## 5. The frozen process: attach to it and read what it is waiting on

So far we have traced programs we launched. The more dramatic use is attaching to a process that is *already running* and, specifically, one that is **stuck** — pinned at zero percent CPU, producing no logs, accepting no requests, just sitting there. Restarting it makes the symptom vanish along with all the evidence. Instead, walk up to it alive:

```bash
strace -p 9120   # attach to PID 9120; Ctrl-C to detach (the process keeps running)
```

For a wedged process, the behavior is itself the diagnosis. strace attaches, prints any in-flight syscall, and then **the output just stops** — because the process is blocked inside a single syscall that has not returned. That last line is the answer. Figure 7 shows the three shapes it usually takes.

![A flow diagram showing a frozen process being attached with strace, the trace hanging on one of three blocking syscalls — futex, read, or poll — each pointing to a deadlock or a dead socket root cause](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-7.png)

**Stuck in `futex`.** You attach and see:

```bash
strace: Process 9120 attached
futex(0x55a3c1e2b9f0, FUTEX_WAIT_PRIVATE, 2, NULL
```

and nothing more — no return value, no newline-completing the call. A `futex` is the kernel primitive underneath a mutex; `FUTEX_WAIT` means the thread is asleep waiting for a lock to be released. If it never wakes, you are almost certainly looking at a **deadlock** (the thread holding the lock is itself waiting on a lock this thread holds — a lock-order inversion) or a **lost wakeup** (the lock was released but the wake never reached this waiter). The trace alone tells you it is a lock problem, not a CPU problem, not an I/O problem. To go further — *which* lock, held by *which* thread — you would now reach for `gdb -p 9120` and dump every thread's stack, which is the technique in the [interactive debugger deep-dive](/blog/software-development/debugging/mastering-an-interactive-debugger). The strace is the thirty-second triage that tells you *which* deeper tool to reach for.

**Stuck in `read`.** You attach and see:

```bash
read(7, 
```

hanging forever. File descriptor `7` is a socket or a pipe, and the peer on the other end is never going to send anything. This is the classic missing-timeout bug: the program did a blocking `read` on a network connection without setting a read timeout, the remote service died or the network dropped the connection in a way that left no FIN, and now the program will wait until the heat death of the universe. You can find out what fd 7 *is* by reading `/proc/9120/fd/7` (a symlink that tells you whether it is a socket, a pipe, a file, and which one). The fix is a read timeout; the trace told you the disease is "blocking read with no timeout," which is a different bug from a deadlock even though both present identically as "frozen."

**Stuck in `poll`/`epoll_wait`/`select`.** An event-loop program (Node, nginx, anything async) that is frozen will be parked in `epoll_wait` waiting for an event that never arrives:

```bash
epoll_wait(4, 
```

This is subtler, because `epoll_wait` is *supposed* to block — that is the normal idle state of an event loop. The diagnosis is whether it is *stuck* or *idle*: an idle loop wakes periodically; a stuck one never does. You confirm by adding `-T` and watching whether the `epoll_wait` ever returns, or by checking whether the program has work queued that it should be processing. A genuinely stuck event loop usually means a single callback ran forever (a synchronous CPU-bound operation that blocked the loop) or a descriptor was never registered with the epoll set.

The general principle for any frozen process: **the blocking syscall is the diagnosis.** `futex` means a lock. `read`/`recvfrom` means a peer that went silent. `poll`/`epoll_wait`/`select` means an event that never fired. `wait4` means a child that never exited (a shell-out that hung, a subprocess waiting on its own input). `flock`/`fcntl` with `F_SETLKW` means a file lock another process holds and will not release — the cause of many "the cron job is stuck" mysteries, where two copies of the job overlap and the second blocks forever on the first's lock file. `accept` means a server waiting for a connection that, if it never comes, points at a listener that lost its backlog. You do not have to reason about the program's logic at all; you read the name of the syscall it is sleeping in and that names the category of bug. From there you escalate to a debugger or to `/proc` for the specifics.

When you need more than the syscall name, the `/proc` filesystem is the natural next stop and it costs the process nothing — reading `/proc` does not stop or slow the process the way `ptrace` does. Three files are worth knowing by heart. `/proc/<pid>/status` shows the process `State` field: an `S` (interruptible sleep) confirms it is blocked in a syscall, a `D` (uninterruptible sleep) is the dangerous one — the process is stuck in the kernel waiting on I/O (usually disk or a stuck NFS mount) and *cannot even be killed* with a normal signal, which is itself a strong diagnosis (your storage layer, not your code). `/proc/<pid>/wchan` literally names the kernel function the process is sleeping in. And `/proc/<pid>/fd/` lists the descriptors, so when the trace shows `read(7)` you run `ls -l /proc/<pid>/fd/7` and see exactly which socket or file fd 7 is — turning the anonymous `7` into `socket:[8453219]` or `/var/log/app.log`. The combination — strace names the syscall, `/proc` names the resource — is the complete picture, and neither step requires you to understand a line of the program's source.

#### Worked example: the cron job that "ran forever"

A nightly batch job that normally finished in twenty minutes was found, one morning, still running after nine hours, holding up the reports that depended on it. CPU was at zero. The on-call engineer's first instinct was to kill and rerun, but killing it blind risked leaving the batch half-applied. Instead: `cat /proc/<pid>/status` showed `State: S (sleeping)` — blocked, not crashed, not spinning. Then `strace -p <pid>` printed one line and hung: `fcntl(3, F_SETLKW, {l_type=F_WRLCK, ...})` — a blocking wait for a write lock on file descriptor 3. `ls -l /proc/<pid>/fd/3` showed fd 3 was `/var/run/batch.lock`. The job was waiting forever for a file lock that *another copy of the same job* held — a previous night's run had wedged on a network mount (its own `State` was `D`, uninterruptible, stuck on the dead NFS server), never released the lock, and tonight's run dutifully queued behind it. The real bug was the original `D`-state process and the dead mount; tonight's "stuck" job was a symptom. The fix was to recover the NFS mount (which freed the original process, which released the lock, which unblocked the queue) and to add a lock timeout so a future overlap would fail fast and loudly instead of hanging silently. Total triage time: about three minutes, all of it `strace -p`, `cat /proc/.../status`, and `ls -l /proc/.../fd`. No debugger, no source.

A practical caution about attaching in production: `strace -p` on a live process inserts the two-context-switch tax we discussed into a process that may be latency-sensitive, and on a busy process unfiltered it can noticeably slow the service while you are attached. For a *frozen* process this is harmless (it is doing nothing anyway). For a busy one, attach with a tight filter and detach quickly, or — better — use `bpftrace`, which we get to in §9.

## 6. The "randomly slow" service: count syscalls, not lines

There is a different flavor of bug where the program is not stuck and not erroring — it just does *too much*, or does the right thing too slowly, and you cannot tell what from reading per-call output because there is too much of it scrolling by. For this you do not want the firehose of individual syscalls; you want the **summary**. That is `strace -c`:

```bash
strace -c -f -p 4815    # attach to a running service, accumulate counts, Ctrl-C to print summary
# or, for a program you launch and that exits:
strace -c -f ./batch_job
```

`-c` suppresses the per-call lines entirely and instead accumulates a table; when you detach (or the program exits) it prints a sorted summary like this:

```bash
% time     seconds  usecs/call     calls    errors syscall
------ ----------- ----------- --------- --------- ----------------
 71.43    4.821907          24    200318           stat
 14.02    0.946612          12     78901           openat
  8.10    0.546800          18     30377     30377 access
  ...
------ ----------- ----------- --------- --------- ----------------
100.00    6.751319                   ...
```

Read this and the bug jumps out: the program spent **71% of its syscall time in `stat`, across 200,318 calls.** Some loop is calling `stat()` on a file (or a directory full of files) two hundred thousand times. This is the fingerprint of a pathological filesystem-scan bug — re-scanning a directory on every iteration, a template engine that `stat`s every include on every render, a module loader that re-checks every path on the search list for every import. The `errors` column is gold too: that `access` line shows 30,377 calls, *all* of which errored, which means the program is probing for files that do not exist, over and over.

The summary turns "it's slow" into a number and a syscall name, and that pair is usually enough to localize the offending code: search for the syscall's library equivalent (`os.stat`, `File.exists`, `fs.statSync`) near the hot loop. Figure 6 walks the timeline of exactly this kind of investigation for a service that was randomly slow because of DNS rather than `stat`.

![A timeline showing a randomly slow service investigated through a strace summary that found ninety percent of time in connect, then per-call timing that revealed a dead IP and DNS retries, ending in a fix that dropped p99 from 1200ms to 80ms](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-6.png)

#### Worked example: the service that was "randomly slow"

A payments-adjacent service had a p99 latency of about 1,200 ms that nobody could reproduce on demand — it was fine most of the time and then a request would take over a second, seemingly at random. Profiling the application code (with the kind of CPU profiler covered in [the Python CPU-profiling deep-dive](/blog/software-development/python-performance/cpu-profiling-cprofile-and-finding-the-hot-path)) showed almost no CPU time — the process was *waiting*, not computing, so the profiler saw nothing. Waiting is invisible to a CPU profiler and glaringly visible to a syscall trace.

We attached `strace -c -f -p <pid>` for sixty seconds during a slow window. The summary was unambiguous: roughly **90% of the time was in `connect` and `poll`**, not in `read`/`write` to the database we expected. Switching to `strace -f -tt -T -e trace=network` on a single slow request showed a `connect` to an IP that returned `ETIMEDOUT` after a long wait, followed by a retry to a *different* IP that succeeded. The root cause: a downstream service's DNS record had two A entries, one of which pointed at a host that had been decommissioned. Roughly half the time the client picked the dead IP first, ate a multi-second connect timeout, then failed over to the live one — producing exactly the intermittent, un-reproducible p99 spike. We pulled the stale A record. The p99 dropped from about **1,200 ms to roughly 80 ms**, and the long tail flattened. The whole diagnosis took one `strace -c` (to point at the network) and one `strace -T` (to measure the timeout). The application code was never the problem, which is precisely why staring at it for a week had found nothing.

The honest measurement note: the "90%" came from the `% time` column of the summary, accumulated over a real slow window, and the "1,200 ms to 80 ms" came from the service's own latency dashboard before and after the DNS fix — both are real measurements, not estimates. When you report a before→after like this, anchor it to a counter the trace gave you (the summary's `% time`) and a metric your monitoring already records (p99), so the number is defensible.

## 7. The fd-exhaustion fingerprint: when opens start failing with EMFILE

There is a particular boundary bug that the trace is almost uniquely good at catching, because it manifests as a *change in the return value of a syscall over time* rather than a single bad call: file-descriptor exhaustion. Every open file, every socket, every pipe consumes a file descriptor, a small integer the kernel hands back. The kernel caps how many a process may hold at once (the `RLIMIT_NOFILE` soft limit — often 1,024 by default, sometimes raised). A program that opens descriptors and forgets to close them leaks them, and when the count hits the limit, the very next `openat` or `socket` fails — not with a logic error, but with `EMFILE` ("Too many open files"). The cruelty is that the leaking code is usually nowhere near the code that finally fails; some innocent request, the one that happened to be the 1,025th open, takes the error and gets blamed.

Under a trace the fingerprint is unmistakable. Early in the run you see opens succeeding with steadily *climbing* fd numbers — `= 23`, `= 24`, `= 25` — that never come back down because nothing is closing them. Then, abruptly:

```bash
socket(AF_INET, SOCK_STREAM, IPPROTO_TCP) = -1 EMFILE (Too many open files)
openat(AT_FDCWD, "/var/lib/app/cache", O_RDONLY) = -1 EMFILE (Too many open files)
```

Two facts jump off those lines. First, the `errno` is `EMFILE`, which only ever means fd exhaustion — there is no ambiguity, no other cause. Second — and this is the diagnostic move — the fd numbers in the successful opens before the failure were *climbing without any matching `close`*. To confirm a leak you trace the open and close calls together and watch whether they balance:

```bash
# Watch opens and closes; in a healthy program these stay roughly balanced over time.
strace -f -e trace=openat,socket,close -e signal=none -p <pid> 2>&1 | grep -E 'openat|socket|close'
```

If you count far more `openat`/`socket` returning new fds than `close` calls retiring them, the gap is your leak, and the trace just told you *which kind* of descriptor is leaking (files? sockets? pipes?) by showing you which syscall opened them. You can cross-check the live count against `ls /proc/<pid>/fd | wc -l`, which lists every descriptor the process currently holds; watching that number climb over minutes while throughput stays flat is the same leak from the other side. This is the trace's view of a problem that the dedicated post on [resource leaks of file descriptors, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) treats in full; here the point is narrower and worth holding onto: **a syscall whose return value degrades over the lifetime of the process — opens that succeed at first and fail later — is the signature of a resource leak, and the trace makes the leak visible long before the process falls over.**

#### Worked example: the connection pool that "randomly" stopped accepting work

A service handled a few hundred requests a second comfortably for hours, then — usually six or seven hours into a run, always at no particular load — began rejecting work with a generic "internal error." Restarting fixed it for another six hours. The six-hour cadence and the load-independence are the tells of a slow leak: something accumulates at a fixed rate per request until it crosses a threshold. We could not afford to `strace` the busy production process, so we did two cheap things. First, `ls /proc/<pid>/fd | wc -l` on a node that had been up five hours: it read about 9,800 open descriptors against a soft limit of 10,240, and on a fresh node the same count was about 200. The descriptor count was climbing at roughly 1,600 per hour and would cross the limit at around the six-hour mark — matching the symptom exactly. Second, we attached `strace -f -e trace=socket,close -p <pid>` for thirty seconds and counted: dozens of `socket(...) = <fd>` calls, almost no `close`. The sockets were outbound connections to a downstream API; a code path created a fresh HTTP client per request instead of reusing a pooled one, and the clients were never closed, so each request leaked one socket. The fix — reuse a single pooled client — dropped the steady-state descriptor count from "climbing toward 10,240" to a flat ~210, and the six-hour crash stopped recurring across a two-week watch. The numbers that made this defensible were both directly observed: the fd count from `/proc/<pid>/fd` before and after, and the leak rate computed from two timestamps. Nothing was estimated.

## 8. ltrace, dtruss, and the macOS reality

Two important neighbors round out the toolkit, and ignoring them will leave you stuck on the wrong platform or one ring too low.

**`ltrace` for the layer above the syscall.** Sometimes the syscall trace is *correct* and the bug is upstream of it — the program computed the wrong argument before ever making the syscall. The config example is the perfect case: by the time `openat` runs, the wrong path is already baked in; what you really want is to see the `getenv`, the `sprintf`, the `strcat` that *built* the path. That is library-call territory, and `ltrace` shows it:

```bash
ltrace -f -e 'getenv+fopen+snprintf' ./app 2>&1 | head
```

You see the library calls and their string arguments — `getenv("APP_CONFIG_DIR")` returning the wrong directory, `snprintf` formatting it into the bad path — which catches the bug a step earlier than strace can. The reality check: `ltrace` is fragile. On binaries built with full RELRO and lazy-binding disabled (now the default on many distros) it cannot plant its PLT breakpoints and falls back to a much slower mode or fails outright; on statically linked or stripped binaries it sees little. Treat it as a sometimes-tool: when it works it is wonderful, and when it does not, drop to `strace` for the syscall view plus `gdb`/`lldb` for the in-process logic.

**macOS: `dtruss`, `dtrace`, and the SIP wall.** There is no `strace` on macOS. The equivalent is `dtruss`, a script built on `dtrace`, Sun's tracing framework that Apple adopted. The invocation is similar in spirit:

```bash
sudo dtruss -f -t open_nocancel ./app     # -t filters to a syscall; -f follows children
sudo dtruss -a -p 9120                     # attach to a running PID
```

But there is a wall you will hit immediately, and it surprises people coming from Linux: **System Integrity Protection (SIP).** SIP, on by default since OS X El Capitan, prevents `dtrace` from tracing any Apple-signed or system binary — you cannot `dtruss` `/usr/bin/python3` as shipped by Apple, you cannot trace most of the system frameworks, and attempts return a permissions error even under `sudo`. You *can* trace your own unsigned binaries and your own builds. The workarounds, in increasing order of severity: trace a binary you compiled yourself rather than a system one; use a Homebrew or pyenv interpreter instead of the system one; or, as a last resort on a dedicated debugging machine, partially disable SIP from recovery mode (`csrutil enable --without dtrace`) — which you should *not* do on a daily-driver or any machine that matters, because it weakens the platform's security guarantees. For most macOS syscall-level debugging, the realistic answer is: build the thing yourself and trace that, or move the reproduction to a Linux container and use `strace`, which is frequently the path of least resistance.

The comparison in Figure 3 lays out all four tracers side by side — what each sees, how it hooks, what it costs, and when to reach for it — so you can pick by the question rather than by habit.

![A matrix table comparing strace, ltrace, bpftrace, and dtruss across what each observes, its hooking mechanism, its overhead, and the situation to reach for it](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-3.png)

| Tool | Layer it sees | How it hooks | Typical overhead | Reach for it when |
| --- | --- | --- | --- | --- |
| `strace` | Syscalls + arguments | `ptrace`, two stops/call | 10–100× on hot processes | One process: wrong path, `EACCES`, hang |
| `ltrace` | Library calls (`malloc`, `getaddrinfo`) | `ptrace` + PLT breakpoints | Often worse than strace; fragile | The bug is in logic *above* the syscall |
| `bpftrace`/bcc | Syscalls, kernel funcs, latency histograms | eBPF probe, in-kernel | Low (percent-level) | System-wide, in production, aggregates |
| `dtruss`/`dtrace` | Syscalls + arguments | DTrace probes | Moderate; **SIP blocks signed binaries** | macOS, on your own unsigned binary |

## 9. bpftrace and eBPF: tracing you can run in production

Everything so far has carried the `ptrace` tax. For development and for one-off triage that tax is fine. For **production**, for **system-wide** questions, and for anything in a hot path, there is a fundamentally better mechanism: **eBPF**. Instead of stopping the process twice per syscall and copying state out to a separate tracer, eBPF lets you attach a tiny, verified program *inside the kernel* that runs at the tracepoint, does its work (increment a counter, record a timestamp, bucket a latency) in kernel space, and aggregates the result in a kernel data structure — no per-syscall context switch out to a userspace tracer, no stopping the process. The overhead drops from "10–100×" to "single-digit percent," which is the difference between a tool you cannot run in prod and one you can. Figure 8 contrasts the two mechanisms directly.

![A two-column before and after diagram contrasting strace adding two context switches per syscall and slowing a hot loop 10 to 100 times against bpftrace running an in-kernel probe that aggregates in a BPF map with percent-level overhead](/imgs/blogs/seeing-what-a-process-really-does-syscall-tracing-8.png)

`bpftrace` is the high-level front end (an awk-like one-liner language); the `bcc` toolkit ships ready-made tools. Here are the one-liners I actually use, each answering a question strace answers only with much more pain in production.

**Count syscalls by process, system-wide:**

```bash
# Which processes are making the most syscalls right now? (the strace -c question, but global)
sudo bpftrace -e 'tracepoint:raw_syscalls:sys_enter { @[comm] = count(); }'
```

Ctrl-C and it prints a histogram of syscall counts keyed by process name. This finds the runaway process without you having to guess which PID to attach to first.

**Trace every file open on the whole machine** — the `opensnoop` tool from bcc, or the bpftrace equivalent:

```bash
sudo opensnoop-bpfcc                 # bcc: prints PID, COMM, FD, ERR, PATH for every open, system-wide
sudo opensnoop-bpfcc -x              # -x: only show FAILED opens (the ENOENT/EACCES ones you care about)
```

`opensnoop -x` is the production-safe version of the config-debugging trick from §3: it shows *every failed open on the system*, so you can catch the wrong-path open even on a process you cannot afford to `strace`. There are siblings: `execsnoop` (every process that execs — catches the surprise subprocess), `tcpconnect`/`tcpaccept` (every outbound/inbound TCP connection with source, destination, and port — the network trick from §4, system-wide and production-safe), `statsnoop` (every `stat`, which finds the 200k-`stat` loop live).

**Latency histogram of a specific syscall** — this is something strace cannot do well at all and bpftrace does beautifully:

```bash
# Distribution of how long read() takes, in microseconds, as a power-of-two histogram
sudo bpftrace -e '
tracepoint:syscalls:sys_enter_read  { @start[tid] = nsecs; }
tracepoint:syscalls:sys_exit_read /@start[tid]/ {
    @us = hist((nsecs - @start[tid]) / 1000);
    delete(@start[tid]);
}'
```

This stamps the entry time per thread, computes the elapsed time on exit, and buckets it into a histogram printed on Ctrl-C. The output is an ASCII histogram (printed by the tool, not by you — this is its native output, not a diagram you drew) showing the distribution of `read` latencies, which instantly distinguishes "all reads are fast, one is catastrophically slow" (a fat tail) from "every read is moderately slow" (a shifted bulk). That tail-versus-bulk distinction is exactly what you need to know and exactly what an average hides.

The practical rule that closes the overhead story: **`strace` for a single process you can afford to slow down (development, triage on a non-critical service); `bpftrace`/bcc for production, for system-wide questions, and for latency distributions.** They are complementary, not competitors. You will use strace ten times a day on your laptop and reach for bpftrace the handful of times the problem is in production and the process cannot be paused.

There is one more property of eBPF that matters for a debugger and is easy to undersell: it is *safe by construction*. Before the kernel loads your bpftrace program it runs it through a **verifier** that statically proves the program terminates (no unbounded loops), touches only memory it is allowed to, and cannot crash the kernel. A buggy `strace` invocation can, at worst, slow a process to a crawl; a buggy bpftrace program is *rejected at load time* rather than corrupting the running kernel. That is why operations teams who would never let you attach `gdb` to a production payments process will let you run a bpftrace one-liner against it — the blast radius is bounded by the kernel itself. The trade-off is a learning curve (the verifier is strict and its error messages are famously cryptic) and a kernel-version dependency (older kernels lack many tracepoints), but for production tracing the safety guarantee is worth the friction.

#### Worked example: finding the noisy neighbor with one bpftrace line

A multi-tenant host's disk was saturated and nobody knew which of the dozen processes on it was responsible — the per-container metrics aggregated I/O but did not attribute the *syscalls* driving it, and the offending process spiked and went quiet on a cycle that made it impossible to catch in the act with `strace` (you cannot attach to a process you have not yet identified). One system-wide bpftrace line settled it: `sudo bpftrace -e 'tracepoint:syscalls:sys_enter_write { @[comm] = count(); }'`, left running for thirty seconds across a saturation window. The histogram on Ctrl-C attributed `write` calls by process name, and one process — a logging sidecar misconfigured to flush every line synchronously — accounted for about 1.9 million `write` calls in thirty seconds, roughly 60,000 per second, dwarfing every other process by two orders of magnitude. No single process could have been `strace`d to find this, because the question was *which* process, system-wide, under a load that the `ptrace` overhead would itself have perturbed. The in-kernel counter answered it with percent-level overhead, on a production host, without pausing anything. The fix (batch the flushes) cut the sidecar's writes by about 99% and the disk saturation cleared. The "1.9 million" was read straight off the bpftrace histogram; the "60,000/sec" is that count divided by the thirty-second window — both observed, neither estimated.

## 10. War stories: when the syscall trace was the whole answer

A few real and realistic cases, to ground the technique in incidents rather than toy programs.

**The thundering DNS herd.** A well-documented class of production outage: a service's resolver configuration lists a primary nameserver that becomes unreachable. Because of how the standard resolver retries — try the first server, wait the full timeout (often five seconds), then fail over — *every* lookup on *every* request suddenly costs five seconds before succeeding on the secondary. Under load, threads pile up waiting on DNS, the connection pool exhausts, and a service that should be making sub-millisecond DNS calls grinds to a halt. From the application's own logs this looks like a mysterious global slowdown with no obvious cause. From `strace -tt -T -e trace=network,poll` it is a one-line diagnosis: a `poll` on the resolver socket with a `<5.000>` elapsed time, repeated on every request. This is the exact shape of the §6 worked example, and it recurs constantly because the resolver's failover behavior is unintuitive and invisible from inside the application. The trace makes the invisible timeout visible.

**The Heartbleed read overflow (a different lens).** Heartbleed (CVE-2014-0160) was a bug in OpenSSL's TLS heartbeat handling where the code trusted an attacker-supplied length field and `memcpy`'d up to 64 KB of process memory back to the client — reading far past the buffer it should have. The bug class is a *read overflow*, and the right tool to *find* such a bug is a sanitizer like AddressSanitizer rather than a syscall tracer (sanitizers are a separate post in this series). But the lens syscall tracing adds is on the *exfiltration*: the stolen data leaves the process the only way anything can — through a `write`/`sendto` syscall on the TLS socket. A trace of the network syscalls on a victim process under attack would show oversized `sendto` payloads in response to tiny heartbeat requests — a request-to-response size asymmetry that is itself a signal. I include it as a reminder that **the syscall boundary is also where data leaves**, so tracing the write side catches a class of "where did this data go" questions that source reading cannot, because the data crosses the same narrow gate everything else does.

**The container that "had no network."** A service deployed fine, started fine, and then could reach nothing — every outbound call failed. The application logs said "connection refused." The instinct was to debug DNS, then routing, then the service mesh. The trace ended it in one command: `strace -f -e trace=network` on the process showed `connect(...) = -1 EACCES (Permission denied)` — not `ECONNREFUSED`, not `ETIMEDOUT`, but `EACCES`. `EACCES` on connect is never a network problem; it is a *policy* problem. The container's network policy (or a seccomp profile, or an egress NetworkPolicy in Kubernetes) was blocking outbound connections. The application's catch-all error handler had flattened every network error into "connection refused," erasing the one piece of information — `EACCES` versus `ECONNREFUSED` — that pointed at the real cause. Hours of network debugging avoided by reading one `errno` off the trace. The recurring lesson across these: **the application's error message is a lossy summary of the syscall's `errno`, and the trace gives you the lossless original.**

## 11. How to reach for this (and when not to)

Syscall tracing is one of the highest-leverage debugging tools that exists for boundary bugs, and it is also one of the easiest to misuse. Some decisive guidance.

**Reach for a syscall trace when** the bug is at the boundary between your code and the OS and the source is not telling you the truth: the file that "should" be found, the host you "should" be connecting to, the permission that "should" be granted, the process that is frozen, the program doing too much I/O. The tell is a mismatch between what the code clearly says and what the program clearly does — that gap is exactly what the trace measures. The trace is also the right *first* tool when you do not yet know whether a problem is yours or the system's, because it draws the line between them precisely.

**Do not reach for it when** the bug is purely in your program's logic above the syscall layer — a wrong calculation, a bad branch, an off-by-one in a pure-data transform that never touches the OS. Those produce correct syscalls with wrong *intent*, and the trace will look perfectly clean while the bug laughs at you; that is a job for a debugger or a print in the computation, not a tracer. The trace sees *effects*, not *reasoning*.

**Never blindly `strace` a hot production process and walk away.** The `ptrace` tax can slow a busy process by an order of magnitude or more, and on a latency-critical service that can itself cause an outage — you would be debugging by breaking. If you must trace in production, filter tightly (`-e trace=openat`), attach briefly, detach immediately, and strongly prefer `bpftrace`/bcc, which carry a fraction of the cost. For a *frozen* process the caution does not apply (it is doing nothing), so attaching to a hung PID with `strace -p` is one of the safest, highest-value moves there is.

**Filter before you read.** Unfiltered strace output is a wall of `mmap`, `brk`, `futex`, and `rt_sigprocmask` that buries the three lines you need. Decide the question first — files? network? a specific slow syscall? — and pass the matching `-e trace=` filter. The filter is not just for readability; it is for overhead, because the kernel runs the un-traced syscalls at full speed.

**Match the timing flag to the question.** `-c` when the question is "what is it doing too much of"; `-T` when it is "how long does each call take"; `-tt` when it is "when did this happen, and in what order relative to that log line." Reaching for the wrong flag buries the answer.

**Escalate deliberately.** The trace is often the *triage* layer, not the final answer. `futex` hang → the trace says "lock problem," then `gdb -p` says "*which* lock and held by whom." 200k `stat` calls → the trace says "filesystem-scan loop," then a CPU profiler or a code search finds the loop. Treat the syscall trace as the instrument that tells you which deeper instrument to pick up.

This decision discipline is the same loop the whole series teaches: the trace is a falsification engine. You hypothesize "the program opens `/etc/app/app.conf`"; the trace either confirms it or hands you the exact contradicting fact (`openat("/etc/app.d/app.conf") = -1 ENOENT`). You hypothesize "the slowness is in the database"; the summary either confirms it or redirects you to `connect`. Every trace is a test of a belief, and the belief is usually what was wrong. This composes with the sibling posts on [tracing the network at the packet and protocol level](/blog/software-development/debugging/its-the-network-packet-and-protocol-tracing) (when the truth lives below the syscall, on the wire) and on hunting [resource leaks of file descriptors, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections) (when the syscalls are individually fine but accumulate). For production-scale observability built in from the start rather than bolted on during an incident, see the system-design treatment of [metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design).

## 12. Reading the output like a native: the grammar of a trace line

A short reference, because the technique is only as good as your fluency in reading the lines, and the syntax is dense the first few times. Every strace line has the same grammar:

```bash
syscall_name(arg1, arg2, arg3, ...) = return_value [errno NAME (description)] <elapsed>
```

Worked through on a real line:

```bash
openat(AT_FDCWD, "/etc/app.d/app.conf", O_RDONLY) = -1 ENOENT (No such file or directory) <0.000019>
```

- `openat` — the syscall. `openat` rather than the older `open` because modern libc uses the `*at` variants that take a directory file descriptor.
- `AT_FDCWD` — the directory fd argument; this special value means "relative to the current working directory," i.e., a normal path.
- `"/etc/app.d/app.conf"` — strace decodes string arguments and shows them in quotes (truncating long ones with `...`; pass `-s 256` to see more of a long string, which matters when you are inspecting the contents of a `write`).
- `O_RDONLY` — the flags, decoded from a bitmask into symbolic names; you would see `O_WRONLY|O_CREAT|O_APPEND` for an append-write open.
- `= -1` — the return value. For `openat`, success is a non-negative file descriptor; `-1` is failure.
- `ENOENT (No such file or directory)` — on failure, strace decodes `errno` into its symbolic name and the human description. **This is the single most valuable token on the line.** `ENOENT` (missing), `EACCES`/`EPERM` (permission), `ECONNREFUSED` (nothing listening), `ETIMEDOUT` (no answer), `EADDRINUSE` (port taken), `EMFILE`/`ENFILE` (out of file descriptors — the fingerprint of an fd leak), `EAGAIN`/`EWOULDBLOCK` (would block, normal on non-blocking I/O): learn this vocabulary and half the trace reads itself.
- `<0.000019>` — present only with `-T`: the elapsed wall time of the call in seconds. A `<5.0>` here is always worth a hard look.

Two reading habits worth building. First, **the return value and `errno` are where the answers are** — scan the right side of the `=` first, looking for `-1` and the `errno`, before you read the arguments. A trace full of successful calls is rarely the bug; the bug is usually the one call that returned an error the program ignored. Second, **follow the file descriptors.** A `read(7, ...)` means nothing until you know what fd 7 is; scroll up to the `openat(...) = 7` or `socket(...) = 7` or `connect(7, ...)` that created or used it, or read `/proc/PID/fd/7`. The fd is the thread that ties a sequence of otherwise-anonymous reads and writes back to the concrete file or socket they act on. Tracing a hang on `read(7)` is only useful once you know fd 7 is the connection to the payment gateway.

A last fluency note on `-f` (follow forks). Real services are not one process; they fork workers, spawn subprocesses, run shell-outs. Without `-f`, strace traces only the parent and goes silent the moment the real work moves into a child — you will swear "nothing is happening" while a forked worker does everything. With `-f`, strace follows every `fork`/`clone`/`vfork` and prefixes each line with the PID that made the call, so a multi-process service's full syscall story is visible. For anything that is not a single-threaded single-process toy, `-f` is not optional; make it muscle memory. The same goes for `-tt`/`-T` once you care about *when* and *how long*, not just *what*.

## 13. Prevent: make the boundary observable before the next 3am

Tracing is a reactive instrument — you reach for it when something is already wrong. The deeper win is using what you learn from a trace to make the next boundary bug *not need* a trace, which closes the *prevent* step of the loop. A few habits.

**Never swallow an `errno`.** The container-network war story happened because the application flattened `EACCES`, `ECONNREFUSED`, and `ETIMEDOUT` into one generic "connection failed." That single act of lossy error handling cost hours, because it threw away the exact distinction the trace later had to recover. When you catch a system error, log the *raw* error — the `errno`, the path, the address — not a friendly summary. A log line that reads `failed to open /etc/app.d/app.conf: ENOENT` is a trace you did not have to run; the program told the truth voluntarily.

**Log the resolved path, not the configured one.** The config bug survived because the code logged "loaded configuration" without saying *which file*. One line — log the absolute path the loader actually opened, and whether it succeeded — would have made the bug self-evident. Anywhere your program turns a configured name into a concrete resource (a path, a host, an fd), log the *resolved concrete value*, because the gap between configured and resolved is exactly where these bugs live.

**Set timeouts on every blocking syscall that talks to the world.** The frozen-on-`read` and hung-on-DNS bugs are both "a blocking call with no timeout." A read timeout, a connect timeout, a resolver timeout: each turns an infinite hang (which you can only diagnose by attaching a tracer) into a fast, logged failure (which diagnoses itself). The trace teaches you which calls can block forever; the prevention is to never let them.

**Bake the production-safe traces into your runbook.** `opensnoop -x`, `tcpconnect`, `execsnoop`, and a syscall-count one-liner are cheap enough to keep in the on-call toolbox and run during an incident without fear. Knowing — before the incident — that "ninety percent of slowness questions are answered by `strace -c` or a bpftrace syscall histogram" turns a panicked unknown into a checklist. The best time to learn to read a trace is not at 3am with the service down; it is now, tracing a healthy program to see what *normal* looks like, so that abnormal jumps out when it matters.

And the meta-prevention, the one that compounds: **trace healthy programs on purpose.** Run `strace -c` on a service you understand and look at its syscall profile when it is *working*. Learn what its normal `openat`/`read`/`write`/`futex` ratios are. Then when it misbehaves, you are not reading a trace cold against no baseline; you are spotting the deviation from a profile you already know. A baseline turns "is 78,000 `openat` calls a lot?" from an unanswerable question into an obvious yes. The engineers who diagnose boundary bugs in minutes are not faster readers; they have a baseline in their heads, and the trace just shows them the delta.

## War story: the 30-second triage that saved a midnight rollback

One more, because it is the pattern in miniature. A deploy went out at 11pm; within minutes a fleet of workers stopped processing the queue. CPU near zero, no errors in the logs, no obvious cause in the diff. The on-call instinct — and the runbook — said roll back. But a rollback at midnight on a thin change is its own risk, and nobody understood *why* it broke, which meant the rollback might not even fix it.

Instead: pick one wedged worker, `strace -p <pid>`. The output printed one line and stopped:

```bash
futex(0x7f9c2a0040a0, FUTEX_WAIT_PRIVATE, 2, NULL
```

Frozen in a `futex` — a lock. Not a CPU problem, not a network problem, not the new code's logic; a deadlock. Thirty seconds of triage had reclassified the incident from "mysterious total failure" to "lock-order bug." `gdb -p` on the same worker dumped the thread stacks and showed two threads in a classic lock-order inversion introduced by the deploy: a new code path acquired lock A then lock B, while an existing path held B and waited on A. The deploy *did* cause it, but the rollback would have been a blind fix; now it was an understood one, and the real fix (order the lock acquisitions consistently) shipped the next morning with a test that would have caught it. The trace did not fix the bug. It did something more valuable at midnight: it told the on-call engineer, in thirty seconds and with certainty, *what kind* of bug it was, which turned a panicked guess into a deliberate decision. That is the whole value proposition of syscall tracing in one incident — it converts "I have no idea" into "it's a lock," fast, from the outside, without restarting and losing the evidence.

## Key takeaways

- **Every effect a program has crosses the syscall boundary, and the kernel records the truth there.** Your source shows intent; the trace shows behavior. Most boundary bugs are exactly the gap between the two.
- **`strace -f -e trace=openat` finds the wrong-path bug in seconds.** The "config isn't being read" mystery is almost always a single `openat` returning `ENOENT` on a path you would never have guessed from the source.
- **The `errno` is the most valuable token on every line.** `EACCES` (policy), `ECONNREFUSED` (nothing listening), `ETIMEDOUT` (no answer), `ENOENT` (missing), `EMFILE` (fd exhaustion) each point at a *different* root cause that the application's generic error message erased.
- **For a frozen process, the blocking syscall is the diagnosis.** `futex` = a lock; `read`/`recvfrom` = a silent peer; `poll`/`epoll_wait` = an event that never fired. Attach with `strace -p`, read the one line it hangs on, escalate to `gdb` for the specifics.
- **`strace -c` turns "it's slow" into a syscall name and a count.** Ninety percent of the time in `connect`, two hundred thousand `stat` calls — the summary localizes the offending loop without reading a single per-call line.
- **`ptrace` costs two context switches per syscall, so always filter and never blindly trace a hot production process.** A busy process can run 10–100× slower under unfiltered strace.
- **For production and system-wide questions, use `bpftrace`/bcc, not `strace`.** In-kernel eBPF probes carry percent-level overhead, and `opensnoop -x`, `tcpconnect`, and latency histograms answer the same questions safely on a live service.
- **macOS uses `dtruss`/`dtrace`, and SIP blocks tracing Apple-signed binaries.** Trace your own builds, or move the reproduction to a Linux container.
- **Prevent the next one: log the resolved path and the raw `errno`, set timeouts on every blocking call, and learn a healthy program's trace baseline** so the abnormal jumps out.

## Further reading

- `strace(1)` man page and the [strace project documentation](https://strace.io/) — the authoritative reference for every flag, including `-e trace=` filter sets, `-y` (print fd paths inline), and `-k` (syscall stack traces).
- Brendan Gregg, [*BPF Performance Tools*](https://www.brendangregg.com/bpf-performance-tools-book.html) and his bpftrace/bcc one-liner collections — the canonical source for production-safe eBPF tracing, `opensnoop`/`execsnoop`/`tcpconnect`, and latency histograms.
- The [bcc and bpftrace repositories](https://github.com/iovisor/bcc) — ready-made tools (`opensnoop`, `statsnoop`, `tcpconnect`, `funclatency`) you can run today, with source you can read to learn the eBPF patterns.
- The Linux `ptrace(2)` and `syscalls(2)` man pages — how the tracing primitive works and the full table of syscalls a trace can show you.
- David Agans, *Debugging: The 9 Indispensable Rules* — Rule 3, "Quit Thinking and Look," is exactly the discipline a syscall trace enforces.
- Within this series: the [scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) (the observe→falsify loop this post serves), [mastering an interactive debugger](/blog/software-development/debugging/mastering-an-interactive-debugger) (where you escalate after the trace names the bug class), and the siblings on [network packet and protocol tracing](/blog/software-development/debugging/its-the-network-packet-and-protocol-tracing) and [resource leaks of fds, sockets, and connections](/blog/software-development/debugging/resource-leaks-fds-sockets-and-connections).
- For building observability in from the start rather than tracing during an incident: the system-design treatment of [metrics, logs, and traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design).
