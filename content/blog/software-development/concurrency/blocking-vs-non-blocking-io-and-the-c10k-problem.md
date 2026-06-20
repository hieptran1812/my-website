---
title: "Blocking vs Non-Blocking I/O and the C10k Problem"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Why a blocking read parks a whole thread, why ten thousand threads run you out of memory before they run you out of CPU, and the readiness-based escape route that quietly powers every modern server."
tags:
  [
    "concurrency",
    "parallelism",
    "non-blocking-io",
    "c10k",
    "event-loop",
    "epoll",
    "scalability",
    "async",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-1.png"
---

A few years ago I watched a perfectly healthy-looking chat server fall over at exactly the wrong moment. It was a simple thing: a TCP server that accepted a connection, spawned a thread to handle it, and read messages in a loop. It worked beautifully in the demo, sailed through the load test at a thousand connections, and shipped. Then a launch happened, traffic climbed past five thousand simultaneous connections, and the process died — not slowly, not with a graceful degradation, but with the operating system's out-of-memory killer reaching in and tearing it down. The CPU graph at the moment of death was almost flat. The cores were *bored*. We had not run out of compute; we had run out of memory, and we had run out of it because every one of those five thousand connections was holding a thread, and every thread was holding a megabyte of stack, and a megabyte times five thousand is five gigabytes of address space dedicated to threads that were, at any given microsecond, almost all of them asleep.

That is the whole story of this post in one paragraph, and it is the reason async exists. A thread that calls a *blocking* read on a socket gets parked by the kernel — frozen, descheduled, removed from the run queue — until bytes arrive. While it is parked it does no work, but it does not give anything back either: it still owns its stack, it still occupies a slot in the scheduler's bookkeeping, it still counts against your memory budget. When most of your connections are idle most of the time — which is the normal state of affairs for a chat server, an API gateway, a websocket fanout, a long-poll endpoint — you end up paying for thousands of parked threads to sit there holding memory while waiting for the network. The thread-per-connection model spends memory like it is free, and it is not.

![thread per connection needs ten thousand threads and about ten gigabytes of stacks and dies from out of memory while one event loop holds ten thousand sockets on a single thread](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-1.png)

This post opens the async track of the series. Everything that follows — the event loop, the reactor pattern, coroutines, async/await, structured concurrency — exists to solve the problem I just described, which has a name: the **C10k problem**, the challenge of handling ten thousand concurrent connections on one machine. We are going to build the problem up from the metal: what a blocking syscall actually does to a thread, what a thread actually costs, why ten thousand of them is untenable, what a *non-blocking* socket is and the `EAGAIN`/busy-poll trap it opens, the difference between *readiness* and *completion* notification, and Little's law — the one piece of math that tells you how much concurrency you actually need to hold. This connects directly to the series' spine: name what is shared, order the accesses, pick the cheapest mechanism. Here the "shared resource" is the machine itself — its memory and its cores — and the question is which concurrency model lets one box hold tens of thousands of mostly-idle connections without falling over. If you have not read [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), that is the framing; and the mechanics of threads and the OS scheduler are in [processes, threads, and how the OS scheduler runs them](/blog/software-development/concurrency/processes-threads-and-how-the-os-scheduler-runs-them), which this post leans on.

## What a blocking syscall actually does to a thread

Let us be precise about the word *blocking*, because it is doing a lot of work and most people use it as a vibe rather than a mechanism. When your code calls `read(fd, buf, n)` on a socket in the default mode, you are making a system call — a controlled transition into the kernel. The kernel looks at the socket's receive buffer. If there are bytes already sitting there (they arrived earlier, the NIC's driver copied them in, the TCP stack acknowledged them), the kernel copies up to `n` of them into your buffer and returns immediately. That is the fast path, and on the fast path "blocking" never blocks.

But if the receive buffer is empty — the peer has not sent anything yet, or the network is in flight — the blocking read does something specific and consequential. The kernel marks your thread as **not runnable**. It changes the thread's state from `TASK_RUNNING` to a sleeping state (`TASK_INTERRUPTIBLE` on Linux), adds the thread to a *wait queue* associated with that socket, and then calls the scheduler to pick somebody else to run on this core. Your thread is now *parked*. It is consuming zero CPU. It will not be considered for execution again until something wakes it.

What wakes it? The data arriving. When a packet for that socket lands at the network card, the NIC raises a hardware interrupt; the driver runs, the TCP/IP stack processes the segment, appends the payload to the socket's receive buffer, and — crucially — walks the socket's wait queue and marks every thread sleeping on it as runnable again (`wake_up`). Your thread goes back into the scheduler's run queue. Some microseconds later the scheduler picks it, restores its registers and stack pointer, and the `read` call finally returns with the bytes. From your code's point of view, `read` just "took a while." From the system's point of view, your thread was asleep for the entire duration and a whole chain of kernel machinery ran to put it to sleep and wake it back up.

![a blocking read calls read then finds no data so the thread is parked idle holding its stack until a packet arrives and the kernel wakes the thread and the read returns after a short run](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-2.png)

Here is the load-bearing fact: **for I/O-bound work, the parked time dominates the running time, often by orders of magnitude.** A request handler might spend forty microseconds of CPU actually parsing and building a response, and then ten milliseconds parked waiting for the next chunk of input from a slow client or a downstream service. That is a 250:1 ratio of waiting to working. The thread is the unit you are paying for, and it is "busy" — in the sense of being unavailable for other work — for the entire ten milliseconds even though it ran for forty microseconds. This is the precise sense in which I/O-bound work *wants concurrency without threads*: you want thousands of these waits in flight at once, but you do not want thousands of threads, because the wait is cheap (the kernel is just holding a wait-queue entry) while the thread is expensive (it holds a stack).

Let me make the mechanism concrete with the simplest possible blocking server, in Go, where the runtime hides a lot of this but the *shape* is exactly thread-per-connection:

```go
// Blocking, one goroutine per connection. Clean and obvious -- and
// the model that does not survive C10k if goroutines were real OS threads.
func serve(ln net.Listener) {
    for {
        conn, err := ln.Accept() // blocks until a client connects
        if err != nil {
            continue
        }
        go handle(conn) // one handler per connection
    }
}

func handle(conn net.Conn) {
    defer conn.Close()
    buf := make([]byte, 4096)
    for {
        n, err := conn.Read(buf) // BLOCKS here until bytes arrive
        if err != nil {
            return // EOF or error: client went away
        }
        conn.Write(buf[:n]) // echo it back (also can block)
    }
}
```

The `conn.Read(buf)` line is where the parking happens. In a language with real OS threads — Java before virtual threads, C with `pthreads`, classic Python threads — that `Read` parks the OS thread, and you need one OS thread per connection to keep this structure. Go cheats: goroutines are *not* OS threads (the runtime multiplexes many goroutines onto a few OS threads, and when a goroutine blocks on network I/O the runtime quietly hands the OS thread to another goroutine — it is using non-blocking I/O underneath, exactly the technique we are building toward). But the *programming model* above is thread-per-connection, and the cost question is: what does one of those handlers cost if it is backed by a real OS thread?

## The thread-per-connection model and its real cost

The thread-per-connection model is the most natural server design in the world. Accept a connection, dedicate a thread to it, write straight-line blocking code: read a request, process it, write a response, loop. The control flow is linear and obvious; there is no callback hell, no state machine, no inversion of control. For decades this was *the* way to write a server, and for low connection counts it is still completely fine. The trouble is purely one of scale, and the scale where it breaks is lower than people expect.

Let us add up what one thread costs. The biggest line item is the **stack**. Every thread needs its own call stack, and the operating system reserves a fixed chunk of address space for it up front. On Linux the default `pthread` stack size is 8 MB of *reserved* virtual address space (`ulimit -s` usually shows 8192 KB), of which physical RAM is only consumed for the pages you actually touch — but you reliably touch a megabyte or more under any real call depth, and the reservation itself eats into your address space and your overcommit budget. The JVM defaults to around 512 KB to 1 MB per thread (`-Xss`). The practical number people quote — and the one I will use throughout — is **about 1 MB of real memory per thread** once you account for the stack pages you touch plus kernel structures. It is an order-of-magnitude figure, not a precise one, and the precise value depends on your platform and your call depth; I am being explicit about that because the kit demands honesty and because the conclusion does not need three significant figures.

On top of the stack there is the **kernel's per-thread bookkeeping**: the task structure (`task_struct` on Linux) is a few kilobytes, plus a kernel-mode stack of its own (typically 8 or 16 KB per thread), plus entries in the scheduler's data structures. None of that is huge per thread, but it is not free, and it is multiplied by your thread count.

![the cost of one blocked thread is about a megabyte of stack plus a few kilobytes of kernel task struct plus a scheduler run slot plus a context switch tax of a few microseconds plus cache footprint adding up to idle but costly](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-4.png)

Then there is the **scheduler cost**, which bites you even when threads are idle but bites hardest when they are not. The OS scheduler has to track every runnable thread and pick among them. With a handful of threads this is trivial; with tens of thousands of runnable threads the scheduler's job — and the cost of every wakeup, which must insert a thread back into the runnable set — grows. And every time the scheduler switches from one thread to another, you pay a **context switch**: save the outgoing thread's registers, switch the page-table root if you are crossing into a different process (you usually are not within one server, but you still flush some state), load the incoming thread's registers, and — the part that hurts most — suffer the cache and TLB pollution as the new thread's working set displaces the old one. A bare context switch is on the order of **1 to 5 microseconds** of direct cost, but the indirect cost from cold caches can be several times that. (The mechanics of the context switch are covered in [processes, threads, and how the OS scheduler runs them](/blog/software-development/concurrency/processes-threads-and-how-the-os-scheduler-runs-them) — here we only need the price tag.)

#### Worked example: the memory wall, not the CPU wall

Take a websocket server holding **10,000** mostly-idle connections — a live dashboard, a chat room, a notifications feed. In the thread-per-connection model that is 10,000 threads. At roughly 1 MB of real memory each, that is **about 10 GB** of RAM consumed by thread stacks alone, before you have stored a single byte of application data. A machine with 8 GB of RAM is already dead. A machine with 16 GB is teetering and will fall over the moment the application's own heap grows. And remember: at this point the CPU is nearly idle, because all 10,000 connections are waiting on the network. You have hit a **memory wall while standing in front of an empty CPU**. That is the signature failure of thread-per-connection, and it is exactly what killed the chat server in the intro at around 5,000 connections — the process ran out of address space and overcommit headroom long before it ran out of compute. The lesson is not "threads are bad." The lesson is that *the thread is the wrong unit of concurrency for a connection that spends 99% of its life waiting.*

There is a second, subtler cost that shows up under churn rather than steady state: thread *creation and teardown* is not free either. Creating a thread means allocating and mapping its stack and registering it with the kernel — tens of microseconds at best. A server that spawns a fresh thread per request (rather than per connection) and tears it down afterward can spend a meaningful fraction of its time just managing threads. This is why thread *pools* exist, and we will come back to them as one of the two escape routes.

The same thread-per-connection shape, written in Java where the thread is unambiguously a real OS thread, makes the cost visible in the code itself. There is no goroutine runtime quietly multiplexing for you; `new Thread(...)` is a genuine kernel thread with a genuine stack:

```java
// Thread-per-connection in Java: every connection gets a REAL OS thread.
// Readable straight-line code -- and ~1 MB of stack per connection.
ServerSocket server = new ServerSocket(8080);
while (true) {
    Socket conn = server.accept();        // blocks until a client connects
    Thread t = new Thread(() -> handle(conn));
    t.start();                            // one kernel thread per connection
}

static void handle(Socket conn) {
    try (var in = conn.getInputStream();
         var out = conn.getOutputStream()) {
        byte[] buf = new byte[4096];
        int n;
        while ((n = in.read(buf)) > 0) {  // BLOCKS this thread until data
            out.write(buf, 0, n);
        }
    } catch (IOException e) {
        // client disconnected
    }
}
```

This code is correct and clear, and at a hundred connections it is a fine server. At ten thousand connections it tries to create ten thousand OS threads, and on a typical box it will either fail to create them (hitting the thread-count or memory limit) or create them and get OOM-killed. The interesting modern wrinkle is **Project Loom's virtual threads** (Java 21+): `Thread.ofVirtual().start(...)` gives you the *same* straight-line blocking code, but the threads are now cheap user-mode "virtual" threads that the JVM multiplexes onto a small pool of carrier OS threads — when a virtual thread blocks on I/O, the JVM parks the *virtual* thread (a few hundred bytes of heap state) and frees the carrier to run another. That is the same trick Go's runtime plays with goroutines: keep the blocking programming model, but make the blocking unit cheap by implementing it on top of non-blocking I/O underneath. Virtual threads are the runtime *hiding* the C10k solution behind a blocking-looking API. Knowing the mechanism in this post is exactly what lets you reason about why a virtual thread is cheap and a platform thread is not — the difference is entirely whether "blocking" parks a megabyte-stack OS thread or a kilobyte-state user thread.

## The C10k problem, stated

In 1999 Dan Kegel wrote an essay titled "The C10k problem," posing a question that sounds quaint now and was provocative then: can a single commodity server handle **ten thousand simultaneous clients**? The hardware of the day could clearly push the bytes — ten thousand connections at a few kilobits each is not a lot of bandwidth — but the *software models* of the day could not. The dominant model was thread-per-connection (or process-per-connection, which was even worse — Apache's classic `prefork` MPM forked a whole process per connection, and a process is heavier than a thread). Kegel's essay catalogued the alternatives — non-blocking I/O with `select`/`poll`, the then-new `epoll` and `kqueue`, asynchronous I/O — and effectively kicked off the modern era of event-driven servers. nginx, Node.js, Netty, libevent, and every async runtime since are answers to the question Kegel posed.

State the problem crisply. You want **C** concurrent connections on **one machine**. The thread-per-connection model costs roughly **C megabytes** of memory (one stack per connection) plus a scheduler that must manage **C** threads. At C = 1,000 that is ~1 GB and a manageable thread count — fine. At C = 10,000 that is ~10 GB and ten thousand threads — untenable on commodity hardware. At C = 100,000 it is absurd. The cost grows **linearly in C**, and the constant (a megabyte) is large. Meanwhile the *actual work* — the bytes to shuffle, the requests to parse — might be tiny, because most connections are idle. You are paying linear memory for connections that are doing nothing.

![C10k math showing one thousand connections cost about a gigabyte of stacks in the thread model but ten megabytes on an event loop, ten thousand connections cost about ten gigabytes and out of memory versus forty megabytes, and one hundred thousand connections are untenable versus two hundred megabytes](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-5.png)

The escape is to **break the one-to-one mapping between connections and threads.** A connection does not need a thread; it needs a little bit of state (its socket, its parse buffer, where it is in the protocol) and a way to be *resumed* when its socket has data. The state is small — a few kilobytes, not a megabyte. The "way to be resumed" is the readiness-notification machinery we are about to build. If one thread can ask the kernel "which of these ten thousand sockets has data right now?" and then service exactly those, you can hold ten thousand connections on **one thread** with **kilobytes** of state each. The memory goes from C megabytes to C kilobytes — a thousandfold reduction — and the thread count goes from C to one (or one per core). That is the C10k solution in a sentence, and the rest of this post is about how you get there.

There is a second, quieter wall in the thread model that is worth naming because it bites before the memory wall on some systems: **file descriptor limits**. Every connection is a file descriptor, and the OS caps how many a process may hold (`ulimit -n`, often defaulting to 1024 on older systems and a few thousand on modern ones). You can raise it — to a million if you want — but the default is frequently the *first* thing that stops a naive server at exactly 1024 connections, before memory even becomes the issue. Both models must raise this limit to reach C10k; the difference is that the event loop *only* needs the fd limit raised, while the thread model needs the fd limit raised *and* enough memory for C stacks *and* a scheduler that can cope with C threads. The event loop removes two of those three walls.

It is also worth being precise about *why* the kernel can do better than user-space polling. When you register a socket with `epoll`, the kernel adds a small callback to that socket's wait queue — the same wait queue a blocking read would have slept on. Now when data arrives and the TCP stack walks the wait queue (exactly as it would to wake a parked thread), instead of waking a thread it runs the `epoll` callback, which adds the socket to a "ready list" inside the `epoll` instance. `epoll_wait` then just hands you that ready list. So the kernel is reusing the *same wakeup machinery* that drives blocking reads, but routing it into a ready list one thread can drain, rather than into ten thousand separate thread wakeups. That is the structural reason readiness notification is O(1) per event: the work is already being done by the network stack; `epoll` just collects it in one place.

It is worth naming that C10k was 1999's number. Today people talk about C10M — ten *million* connections on one box — which requires bypassing even the kernel's per-connection overhead (kernel-bypass networking, user-space TCP stacks like the techniques behind DPDK). The arithmetic is the same; only the constants move. The conceptual leap is identical: stop spending a thread per connection.

## Non-blocking sockets: O_NONBLOCK, EAGAIN, and the busy-poll trap

The first tool in the kit is the **non-blocking socket**. You take a socket and flip a flag — `O_NONBLOCK` — and you change the contract of every I/O call on it. A non-blocking `read` *never parks the thread*. If there is data, it returns it. If there is no data, instead of putting the thread to sleep, it returns immediately with the error code **`EAGAIN`** (equivalently `EWOULDBLOCK` — on Linux they are the same value). `EAGAIN` is not really an error in the "something went wrong" sense; it is the kernel saying "not right now, try again." Likewise a non-blocking `write` that cannot fit your bytes into the socket's send buffer returns `EAGAIN` rather than blocking until there is room.

Here is how you set it, in C, where the mechanism is fully exposed and there is no runtime hiding anything:

```c
#include <fcntl.h>
#include <errno.h>
#include <unistd.h>

// Flip a socket into non-blocking mode: read/write return EAGAIN
// instead of parking the calling thread when they would block.
static int set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    if (flags == -1) return -1;
    return fcntl(fd, F_SETFL, flags | O_NONBLOCK);
}

// A non-blocking read that handles the EAGAIN case explicitly.
ssize_t try_read(int fd, char *buf, size_t n) {
    ssize_t r = read(fd, buf, n);
    if (r == -1) {
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return 0; // "no data right now" -- NOT an error, just not ready
        }
        return -1; // a real error (ECONNRESET, etc.)
    }
    return r; // r > 0 bytes read, or r == 0 means peer closed
}
```

This is genuinely powerful: with non-blocking sockets, one thread can attempt I/O on many sockets without ever getting stuck on any single one. But it opens a trap so seductive that nearly everyone falls into it the first time, and it is worth walking into deliberately so you understand why the next tool is necessary.

The naive way to use non-blocking sockets is to **loop over all your connections and try to read each one, around and around forever**:

```c
// THE BUSY-POLL TRAP. Correct, and a disaster.
// One thread, all connections non-blocking. Looks like it scales!
for (;;) {
    for (int i = 0; i < num_conns; i++) {
        ssize_t r = try_read(conns[i].fd, conns[i].buf, BUF_SIZE);
        if (r > 0) {
            handle_data(&conns[i], r); // got something, process it
        }
        // r == 0 (EAGAIN): nothing to do, move to the next socket
    }
}
```

Read that loop and ask: what does the CPU do when *all* the connections are idle — which, remember, is the normal state? The answer is that it **spins as fast as it possibly can**, ripping through the entire list of connections, calling `read` on each, getting `EAGAIN` from every one, and immediately doing it again. It pegs a CPU core at 100% utilization while accomplishing exactly *nothing*. You have traded the thread-per-connection model's *memory* wall for a *CPU* wall: one thread, one core, burned to the ground busy-polling sockets that have no data. On a multi-tenant box your neighbors hate you; on a laptop your fan screams; in a data center your power bill notices. This is the **busy-poll trap**, and it is the reason non-blocking sockets *alone* are not the answer.

![busy poll loop retries on EAGAIN and burns a full core at one hundred percent CPU doing no work while readiness notification sleeps in epoll wait at near zero idle CPU and wakes only when a socket becomes ready](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-6.png)

#### Worked example: counting the wasted syscalls

Put numbers on the waste. Suppose a `read` syscall that returns `EAGAIN` costs about 100 nanoseconds (a light syscall — kernel entry, check the buffer, return). With 10,000 sockets, one full sweep of the busy-poll loop is 10,000 such calls, roughly 1 millisecond of pure syscall overhead doing nothing useful. With no sleep, the loop runs about 1,000 sweeps per second per core — that is **ten million wasted `EAGAIN` syscalls per second**, one core pinned at 100%, and zero forward progress on idle connections. Now compare the readiness model: `epoll_wait` makes *one* syscall, the thread sleeps consuming no CPU, and it wakes only when (say) 19 sockets out of 10,000 are actually ready — 19 reads instead of ten million. The ratio of useful to wasted work in the busy-poll version is essentially zero, and that is not hyperbole; it is the arithmetic. This is why the busy-poll trap is not a "slightly less efficient" approach you can tune your way out of — it is categorically the wrong shape, and adding a `sleep` only trades CPU waste for latency without fixing the O(N)-per-sweep syscall problem.

You can soften the busy-poll trap by sleeping a few milliseconds between sweeps — `usleep(1000)` at the bottom of the outer loop — and people do exactly that in quick scripts. But now you have introduced latency (up to your sleep interval before you notice new data) and you are *still* doing O(num_conns) useless syscalls every sweep. With ten thousand sockets you make ten thousand `read` calls per sweep just to discover that nineteen of them have data. The fundamental problem is that **you are asking the wrong question.** You are repeatedly asking each socket "do you have data?" when what you want is for the kernel — which already knows, because it is the one putting data into those buffers — to *tell you* which sockets have data. That inversion is the whole game.

The same non-blocking contract exists in every language, and the `EAGAIN` shows up the same way. In Rust, the standard library surfaces "would block" as an error of kind `WouldBlock`, and the idiom is to match on it explicitly rather than letting it propagate as a failure:

```rust
use std::io::{ErrorKind, Read};
use std::net::TcpStream;

// Set the socket non-blocking, then read. WouldBlock is Rust's EAGAIN:
// it is "not ready", not a real failure.
fn try_read(stream: &mut TcpStream, buf: &mut [u8]) -> std::io::Result<usize> {
    stream.set_nonblocking(true)?;
    match stream.read(buf) {
        Ok(0) => Ok(0),                                  // peer closed
        Ok(n) => Ok(n),                                  // got n bytes
        Err(ref e) if e.kind() == ErrorKind::WouldBlock => Ok(0), // EAGAIN
        Err(e) => Err(e),                                // real error
    }
}
```

The single most important pattern once you have a non-blocking socket is the **drain loop**: when you learn a socket is readable, you do not read once — you read *repeatedly until you get `EAGAIN`*, because a single readiness signal may correspond to many kilobytes of buffered data, and especially under edge-triggered notification (which `epoll` offers and which fires *once* per readiness transition) you will hang forever if you stop early and leave bytes unread. The drain loop is the bridge between non-blocking sockets and the event loop, and it looks like this in C:

```c
// Drain a known-ready socket: read until the kernel says EAGAIN.
// Mandatory under edge-triggered epoll, good hygiene under level-triggered.
void drain_socket_nonblocking(int fd) {
    char buf[4096];
    for (;;) {
        ssize_t r = read(fd, buf, sizeof buf);
        if (r > 0) {
            process(buf, r);          // handle this chunk
            continue;                 // there may be more -- keep going
        }
        if (r == 0) {
            close_connection(fd);     // peer closed cleanly
            return;
        }
        if (errno == EAGAIN || errno == EWOULDBLOCK) {
            return;                   // drained: nothing left right now
        }
        if (errno == EINTR) continue; // interrupted by a signal, retry
        handle_error(fd);             // a real error
        return;
    }
}
```

Notice the difference from the busy-poll trap. Here `EAGAIN` is the *exit condition of draining a socket the kernel already told us is ready* — we read greedily until the buffer is empty, then stop and wait to be told again. In the busy-poll version, `EAGAIN` was the answer to a question we should never have asked, because we were guessing instead of being notified. Same error code, opposite meaning, and the difference is whether something woke you up first.

## Readiness versus completion: what the kernel can tell you

The fix is **readiness notification**: instead of polling each socket, you hand the kernel your whole set of sockets once and say "put me to sleep, and wake me when *any* of these becomes ready to read or write — and tell me which ones." The thread sleeps (zero CPU) until there is real work, then wakes with a list of exactly the sockets that have data. No busy-polling, no per-socket guessing, no wasted syscalls. One thread can wait on ten thousand sockets and be woken only for the handful that are active at any instant.

There is a spectrum of mechanisms here, and they matter because their *scaling behavior* differs:

| Mechanism | Cost per wait | Scales to | Notes |
| --- | --- | --- | --- |
| `select` | O(N) scan of all fds | ~1024 fds (hard limit) | the original; rebuilds the fd set every call |
| `poll` | O(N) scan of all fds | thousands | no 1024 limit, still O(N) per call |
| `epoll` (Linux) | O(1) per ready fd | 100k+ | register once, kernel keeps the set, returns only ready fds |
| `kqueue` (BSD/macOS) | O(1) per ready event | 100k+ | the BSD equivalent of epoll |
| IOCP (Windows) | O(1), completion-based | 100k+ | completion model, not readiness |
| `io_uring` (Linux) | O(1), completion-based | 100k+ | submission/completion rings, fewer syscalls |

`select` and `poll` were the first answers, and they share a fatal flaw at scale: every call is **O(N)** in the number of file descriptors you are watching. You pass the whole set in, the kernel scans the whole set, and you scan the whole set again on return to find the ready ones. With ten thousand sockets, every single `epoll_wait`-equivalent call walks ten thousand entries — even if only one is ready. That is why `select`/`poll` do not solve C10k: they replace O(N) busy-polling in user space with O(N) scanning in the kernel, which is better (the thread sleeps) but still linear.

**`epoll`** (Linux, 2002) and **`kqueue`** (BSD) broke the linearity. The idea: you *register* your interest in a set of sockets once, the kernel maintains that set internally and hooks each socket's wait queue, and when you call `epoll_wait` the kernel hands you back *only the file descriptors that are actually ready* — O(1) per ready event, not O(N) over everything. Register ten thousand sockets, and if three are ready, `epoll_wait` returns three. That is the breakthrough that made C10k routine. The next post, [the event loop and the reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern), is entirely about building a server on top of `epoll`/`kqueue`, so I will not go deep here — I just want the *shape* on the table:

```c
// READINESS NOTIFICATION with epoll: register once, sleep, wake on ready fds.
int ep = epoll_create1(0);
struct epoll_event ev = { .events = EPOLLIN, .data.fd = listen_fd };
epoll_ctl(ep, EPOLL_CTL_ADD, listen_fd, &ev); // register interest, once

struct epoll_event events[MAX_EVENTS];
for (;;) {
    // Sleeps here (zero CPU) until at least one fd is ready.
    int n = epoll_wait(ep, events, MAX_EVENTS, -1);
    for (int i = 0; i < n; i++) {     // only the READY fds, not all of them
        int fd = events[i].data.fd;
        if (fd == listen_fd) {
            accept_new_connection(ep, listen_fd); // register the new socket
        } else {
            drain_socket_nonblocking(fd);          // read until EAGAIN
        }
    }
}
```

Notice the structure. The thread spends almost all its time asleep inside `epoll_wait`. It wakes only when the kernel has real news. When it wakes, it processes exactly the ready sockets and goes back to sleep. The non-blocking sockets and `EAGAIN` from the previous section are still essential — you read each ready socket *until it returns `EAGAIN`* so you drain everything available without blocking — but now the `EAGAIN` is the *exit condition of draining a known-ready socket*, not a "nothing happened" you got from blindly guessing. That is the difference between using `EAGAIN` well and falling into the busy-poll trap.

![three ways to wait on a socket compared as a table, blocking read parks the thread and scales to about five thousand threads, non-blocking poll spins and wastes a full core and scales to a few connections, readiness epoll blocks once and wastes near zero CPU and scales to one hundred thousand connections](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-3.png)

There is one more distinction worth pinning down because it shapes the whole API design: **readiness** versus **completion**.

- **Readiness** (`epoll`, `kqueue`, `select`, `poll`): the kernel tells you "this socket is *ready* — you can read/write now without blocking." *You* then perform the actual `read`/`write` yourself, in non-blocking mode, and handle the bytes. The kernel told you *when*; you did the I/O.
- **Completion** (Windows IOCP, Linux `io_uring`, POSIX AIO): you tell the kernel "read N bytes into *this* buffer, and let me know when it's *done*." The kernel performs the I/O itself and notifies you when the bytes are already in your buffer. You never call `read` — you submit a request and collect a completion.

The readiness model dominates Unix server programming (it is what `epoll`/`kqueue` give you, and it is what most event loops are built on). The completion model is native on Windows and is the future on Linux via `io_uring`, which reduces syscall overhead by batching submissions and completions through shared memory ring buffers. For the rest of this post I will assume the readiness model because it is the one underlying the C10k story and the event loop; just know that "the kernel tells me which sockets I can act on" (readiness) and "the kernel does the I/O and tells me it's finished" (completion) are two genuinely different contracts, and the choice ripples through your entire I/O layer.

One more knob deserves a name because it interacts with the drain loop above: **level-triggered** versus **edge-triggered** notification. Level-triggered (the default for `epoll`, and how `select`/`poll` behave) means the kernel reports a socket as ready *as long as* there is data to read — so if you read only part of the buffer and call `epoll_wait` again, it tells you the socket is still ready. Edge-triggered (`EPOLLET`) reports readiness *only on the transition* from not-ready to ready — it fires once when data first arrives and will *not* fire again until *new* data comes, even if you left bytes unread. Edge-triggered is more efficient (fewer wakeups) but unforgiving: it *requires* the drain-until-`EAGAIN` loop, because if you stop reading early you have unread bytes the kernel will never re-notify you about, and that connection silently hangs. This is one of the most common bugs in hand-rolled event loops, and it is why I emphasized the drain loop earlier. The mechanism explains the bug: edge triggering is a one-shot signal on a state *change*, so you must consume all the state the signal told you about, or you lose the news.

## A connection without a thread: the per-connection state object

Here is the conceptual pivot the whole C10k solution turns on, and it is worth slowing down for because it is the thing that makes the memory math work. In the thread model, a connection's *state* lives implicitly in a thread's stack: where you are in the protocol is encoded by *which line of `handle()` the thread is currently parked on*, and the partial data you have read so far sits in local variables on that stack. That is convenient — the language's call stack is doing your bookkeeping for free — but it is *expensive*, because the stack that holds those few bytes of meaningful state is a whole megabyte of reserved space.

The event loop makes that state **explicit and small**. Instead of one stack per connection, you keep one little *struct* per connection — its socket, its read buffer, how many bytes it has accumulated, and an enum saying where in the protocol it is (reading the header, reading the body, writing the response). When `epoll_wait` reports that connection ready, you load its struct, advance its state machine by however much you can do without blocking, save the struct, and move on. The connection's "where am I?" is now a field in a few-kilobyte object on the heap rather than the position of a parked thread inside a megabyte stack. That is the entire memory win, stated structurally: **you replaced an implicit megabyte (the stack) with an explicit few kilobytes (the state object).** Multiply by ten thousand and you have the difference between 10 GB and tens of megabytes.

Concretely, a per-connection state object and its non-blocking advance look like this in Go, using a small enum to track protocol position:

```go
type connState int

const (
    readingHeader connState = iota
    readingBody
    writingResp
    done
)

// One small struct per connection -- a few KB, not a 1 MB stack.
type conn struct {
    fd       int
    state    connState
    buf      []byte // accumulated bytes so far
    needed   int    // how many bytes this phase wants
}

// Advance the connection as far as possible WITHOUT blocking.
// Called when epoll says this fd is readable; returns when it would block.
func (c *conn) advance() {
    for c.state != done {
        n, err := nonblockingRead(c.fd, c.buf[len(c.buf):])
        if err == errEAGAIN {
            return // not enough data yet -- save state, wait for next readiness
        }
        if err != nil || n == 0 {
            c.state = done
            return
        }
        c.buf = c.buf[:len(c.buf)+n]
        if len(c.buf) >= c.needed {
            c.step() // header complete -> body, body complete -> respond, etc.
        }
    }
}
```

The key line is `return` on `errEAGAIN`: the connection saves its partial progress in `c.buf` and `c.state` and yields, costing nothing while it waits, and resumes exactly where it left off the next time `epoll` says it is ready. No thread is parked; the state lives in `c`. This is what "concurrency without threads" *means* mechanically — the waiting is represented by a small saved struct, not by a frozen thread holding a stack. Writing this state machine by hand is tedious and error-prone (that is the cost the event loop imposes), which is precisely why coroutines and async/await were invented: they let the compiler generate this state machine *for* you from straight-line-looking code, so `await read(...)` compiles down to "save my state and yield, resume here when ready." But underneath the syntax, every async runtime is doing exactly what `advance()` does above — saving a small per-task state object on a readiness yield, never parking a thread on a wait.

## Little's law: how much concurrency do you actually have?

Before we pick an escape route, we need to size the problem, and there is exactly one piece of math you need for that. It comes from queueing theory and it is called **Little's law**:

$$L = \lambda W$$

In words: the average number of items *in a system* ($L$) equals the average arrival rate into the system ($\lambda$, items per second) times the average time each item *spends* in the system ($W$, seconds). It is one of those results that is almost embarrassingly simple to state and astonishingly general — it holds for any stable system regardless of the arrival distribution, the service distribution, or the number of servers, as long as things are in steady state (what comes in eventually goes out). For a server, the "items" are in-flight requests or connections, $\lambda$ is your request rate, and $W$ is how long each request takes end to end. Then $L$ — the thing Little's law hands you — is **the number of requests you have in flight at any instant**, which is *exactly the amount of concurrency you must be able to hold simultaneously.*

This is the number that decides everything. Your thread count, your connection cap, your memory budget — they all flow from $L$. And the trap people fall into is sizing for the *rate* ($\lambda$) and forgetting the *duration* ($W$), when it is their *product* that determines how much you hold at once.

![Little's law sizing where an arrival rate of ten thousand requests per second times a service time of one second gives ten thousand requests in flight, which costs ten gigabytes of stacks in the thread model but is held entirely by one event loop](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-7.png)

#### Worked example: rate is a liar, duration is the truth

Two services, same throughput, wildly different concurrency. Service A is a fast cache lookup: it handles $\lambda = 10{,}000$ requests per second, and each request takes $W = 0.001$ seconds (1 ms). By Little's law it has

$$L = \lambda W = 10{,}000 \times 0.001 = 10 \text{ in flight.}$$

Ten. You could serve service A with a tiny thread pool and never feel a thing. Now service B handles the *same* $\lambda = 10{,}000$ requests per second, but each request is a slow downstream call (an external API, a big query, a long-poll) taking $W = 1$ second:

$$L = \lambda W = 10{,}000 \times 1 = 10{,}000 \text{ in flight.}$$

Ten *thousand*. Same arrival rate, a thousandfold more concurrency, purely because each request lingers a thousand times longer. In the thread-per-connection model service B needs ten thousand threads — about 10 GB of stacks — and dies, while service A is comfortable on a handful of threads. **The two services have identical throughput and opposite fates, and the only difference is $W$.** This is why "how many requests per second?" is the wrong first question for sizing concurrency. The right question is "how many are *in flight at once*?" — and that is $\lambda W$.

Now connect it back to the whole post. The reason I/O-bound work explodes your concurrency is that I/O *inflates $W$*. CPU-bound work has a small $W$ (you compute and you are done), so $L$ stays small and a thread pool sized to your core count is perfect. I/O-bound work has a large $W$ (you wait on the network for most of it), so $L$ balloons — and every unit of that $L$ is, in the thread model, a parked thread holding a megabyte. Little's law is the bridge between "my requests are slow because they wait on I/O" and "therefore I have ten thousand of them in flight and cannot afford a thread each." The slow part of $W$ is *waiting*, and waiting is precisely what does not need a thread. You want to hold $L = 10{,}000$ waits with kilobytes each, not $L = 10{,}000$ threads with megabytes each.

#### Worked example: sizing a blocking thread pool the honest way

Suppose you are *not* going async — you have a legacy blocking client library you cannot replace — and you want to size a thread pool for service B. You measured $W = 1$ s per request and you want to sustain $\lambda = 2{,}000$ req/s (not the full 10k; you are scaling out across machines). Little's law says you need $L = 2{,}000 \times 1 = 2{,}000$ threads in flight to hit that rate. So you need a 2,000-thread pool — about 2 GB of stacks — *just for this one service on this one machine*. If $W$ is actually 1.5 s under load (it usually degrades), you need 3,000 threads, or your effective $\lambda$ drops and requests queue up. This is the brutal honesty Little's law forces: a blocking thread pool's size is dictated by $\lambda W$, and when $W$ is large, that number is large, and large thread counts are exactly what we are trying to avoid. The math does not let you cheat. The only way to shrink the thread count *without* shrinking throughput is to make the *waiting* not cost a thread — which is async.

## Two escape routes: bounded thread pool versus event loop

You have a connection count $L$ that is too large to give each connection its own thread. There are exactly two families of solution, and they correspond to a fork in the road: keep blocking but *bound* the threads, or stop blocking and *multiplex* on an event loop.

![two escape routes from C10k as a tree, keep blocking which leads to a bounded thread pool and queue overflow with backpressure, or go non-blocking which leads to an event loop on one thread and async await coroutines](/imgs/blogs/blocking-vs-non-blocking-io-and-the-c10k-problem-8.png)

**Escape route one: the bounded blocking thread pool.** Keep your nice straight-line blocking code, but instead of one thread per connection, use a *fixed* pool of, say, 200 worker threads, and a queue of pending work in front of them. When work arrives, it goes on the queue; a free worker picks it up, blocks its way through it, and returns to the pool. The thread count is now bounded — 200 threads is 200 MB, not 10 GB — and the linear-in-C memory blowup is gone. This is the model behind Java's `ExecutorService`/`ThreadPoolExecutor`, the classic servlet container, and most "synchronous" web frameworks. It is simple, the code is readable, and for many workloads it is entirely correct.

But bounding the pool does not make Little's law go away; it relocates the pressure to the *queue*. If $L = \lambda W$ exceeds your pool size, requests pile up in the queue, latency climbs (each request now waits for a worker *plus* its own service time), and if load keeps rising the queue grows without bound until you run out of memory a different way — or you bound the queue too, and then you must *reject* or *shed* work when it is full. That rejection is **backpressure**, and it is the correct behavior: a bounded pool with a bounded queue degrades gracefully (it sheds load) instead of falling over (it OOMs). The architecture-level treatment of this is in [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure); the point here is that the thread pool *trades the memory wall for a latency-and-backpressure problem*, which is often a much better problem to have. Crucially, the thread pool only helps if $L$ is *moderate*. For 200 in-flight requests, a 200-thread pool is great. For 10,000 in-flight mostly-idle connections, a 200-thread pool means 9,800 connections are stuck in the queue not being read — useless for a websocket fanout. Thread pools are for *bounded concurrency of active work*, not for *holding many idle connections*.

Here is the bounded-pool model in Java, which is its native habitat:

```java
// Escape route one: a BOUNDED thread pool with a bounded queue.
// Memory is capped at poolSize stacks; overflow is rejected (backpressure).
ExecutorService pool = new ThreadPoolExecutor(
    200, 200,                       // 200 worker threads, fixed
    60L, TimeUnit.SECONDS,
    new ArrayBlockingQueue<>(1000), // bounded queue: at most 1000 waiting
    new ThreadPoolExecutor.AbortPolicy() // reject when full -> backpressure
);

void handle(Socket conn) {
    try {
        pool.submit(() -> {
            // straight-line BLOCKING code, the whole appeal of this model
            try (var in = conn.getInputStream();
                 var out = conn.getOutputStream()) {
                byte[] buf = new byte[4096];
                int n;
                while ((n = in.read(buf)) > 0) { // blocks a pool thread
                    out.write(buf, 0, n);
                }
            } catch (IOException e) { /* client gone */ }
        });
    } catch (RejectedExecutionException e) {
        reject(conn); // pool + queue full: shed load instead of OOM
    }
}
```

**Escape route two: the single-threaded event loop.** Make every socket non-blocking, register them all with `epoll`/`kqueue`, and run *one* thread (or one per CPU core) that sleeps in `epoll_wait`, wakes on ready sockets, does a quick non-blocking read/process/write for each, and goes back to sleep. There is no thread per connection — there is *no thread per anything*. Each connection is just a little state object (its socket, its buffer, its protocol position) living on the heap, costing kilobytes. Ten thousand connections is ten thousand small state objects and one busy thread. This is the model behind nginx, Node.js, Netty, Redis, and every async runtime. It collapses the memory cost from $C$ megabytes to $C$ kilobytes and the thread count from $C$ to one. The next post builds this in full.

The catch — and there is always a catch — is that the event loop *inverts your control flow*. You can no longer write "read, then process, then write" as straight-line code, because you must not *block* the one precious thread that is servicing every connection. The moment any handler blocks — on a slow disk read, a synchronous database driver, a `sleep`, a CPU-heavy compute — the *entire* event loop stalls and *every* connection freezes, because they all share that one thread. So event-loop code must be written as a state machine: do the part you can do now, register a callback (or `await` a future) for the part that needs to wait, and return control to the loop. This is where callbacks, promises, futures, and ultimately async/await come from — they are ergonomics layered on top of the event loop to make state-machine code *look* like straight-line code again. The cost of the event loop is not memory; it is *programming model complexity* and the ever-present hazard of accidentally blocking the loop. Here is the trade in one comparison:

| Dimension | Thread per connection | Bounded thread pool | Event loop |
| --- | --- | --- | --- |
| Memory at 10k conns | ~10 GB (dies) | ~200 MB (only 200 active) | ~tens of MB |
| Code style | straight-line blocking | straight-line blocking | state machine / async |
| Holds many idle conns? | yes, but too costly | no (queued, unread) | yes, cheaply |
| Failure mode | OOM | queue grows / sheds load | one blocking call stalls all |
| Best for | low conn count | bounded active work | many concurrent connections |
| Examples | naive servers, old Apache | Java servlets, sync frameworks | nginx, Node, Netty, Redis |

A real production system often uses *both*: an event loop for the I/O (holding the connections cheaply) plus a *bounded* thread pool off to the side for the genuinely blocking or CPU-heavy bits (so they do not stall the loop). Node.js does exactly this — its libuv thread pool handles file I/O and DNS behind the single-threaded event loop. The two escape routes are not enemies; the mature answer is to use the loop for waiting and the pool for working.

## Measured: memory and throughput at 1k and 10k connections

Now the part the kit insists on — measured behavior, honestly framed. I am going to give you the *shape* of the numbers you will see if you run this experiment yourself, with explicit caveats about what is solid and what is an order-of-magnitude estimate, because fabricating precise figures would be worse than useless.

The experiment: a trivial echo server, implemented twice — once thread-per-connection (one OS thread per socket, blocking reads) and once as a single-threaded `epoll` event loop (non-blocking sockets). Drive it with a load generator opening $C$ connections that each send a small message and wait, so the connections are *mostly idle* — the realistic websocket/long-poll shape. Measure resident memory (RSS) and sustained throughput. Warm up first, run several times, and watch for the OS killer.

| Metric | Thread-per-conn @ 1k | Event loop @ 1k | Thread-per-conn @ 10k | Event loop @ 10k |
| --- | --- | --- | --- | --- |
| Threads | ~1,000 | 1 | ~10,000 (if it survives) | 1 |
| RSS (memory) | ~1–2 GB | ~20–40 MB | ~10 GB or OOM | ~40–80 MB |
| Idle CPU | low | ~0% | low (or thrashing) | ~0% |
| Throughput | OK | OK | degraded / dead | steady |
| Outcome | survives | comfortable | often OOM-killed | comfortable |

What is *solid* here: the memory ratio. The event loop uses one to two orders of magnitude less memory than thread-per-connection at the same connection count, because it holds kilobytes of state per connection instead of a megabyte of stack. That gap is real, large, and reproducible — it is the entire point. The thread model's memory grows linearly with $C$; the event loop's barely moves. At 10k connections the thread server is at the edge of or past what a typical 8–16 GB box can hold, while the event loop is using tens of megabytes. This is not subtle.

What is *fuzzier* and platform-dependent: the exact RSS figures (your stack size, your overcommit settings, and your kernel version all move them), the throughput numbers (dominated by your NIC, your kernel, and whether you tuned `somaxconn`, file-descriptor limits via `ulimit -n`, and `net.core` sysctls), and *whether the thread server OOMs or merely thrashes* at 10k (depends entirely on your RAM and `vm.overcommit_memory`). So I will not quote "exactly 9,743 connections before death" — that number would be a lie dressed up as precision. The defensible claim is: **thread-per-connection memory scales as roughly 1 MB × C and falls over in the single-digit-thousands on commodity hardware; the event loop scales as roughly a few KB × C and comfortably reaches 10k–100k.** That order-of-magnitude statement is what you should carry, and it is what every real measurement confirms.

How to run this honestly, if you want to reproduce it rather than trust me: raise the file-descriptor limit on both server and client (`ulimit -n 200000`) so the fd cap is not what stops you — you want to measure the *model*, not a default. Use a load generator that holds connections open (a tool like `wrk` with persistent connections, or a small custom client that opens C sockets and keeps them idle with occasional pings) rather than one that hammers short-lived requests, because the whole point is *many idle held connections*, which is where the thread model dies and a request-per-second benchmark would hide the problem. Warm up — let the JIT compile, let the kernel populate its caches, let the allocator settle — then take several runs and report the spread, not a single number; concurrency benchmarks are noisy and a lone figure is meaningless. Watch `RSS` (in `/proc/<pid>/status` as `VmRSS`, or `ps -o rss`) climbing as you add connections, and watch `dmesg` for the OOM killer on the thread server. The thing you are looking for is not a single throughput number; it is the *slope* — does memory grow linearly with connection count (thread model) or stay flat (event loop)? That slope is the answer, and it is robust across platforms even when the absolute numbers are not.

A subtle point that trips people up: at *low* connection counts the thread model can actually post *better* latency per request than the event loop, because a dedicated thread on a free core handles its connection with no multiplexing overhead and the kernel scheduler is happy to keep it warm. The event loop's advantage is not lower per-request latency at small scale — it is that its cost does not *explode* at large scale. If you benchmark at 100 connections you may see the thread server look great and conclude async is pointless; the async win only shows up when you push past where the thread model's memory and scheduler costs start to dominate. Always benchmark at the connection count you actually need to survive, not a comfortable small one, or you will measure the wrong regime and draw the wrong conclusion.

#### Worked example: reading the failure honestly

When the chat server in the intro died at ~5,000 connections, the diagnosis was a one-liner once you knew to look. `dmesg` showed the OOM killer firing. `ps` showed ~5,000 threads. RSS was climbing linearly with connection count at almost exactly 1 MB per connection — the stack signature. CPU was at 15%. That combination — *memory linear in connections, CPU low, OOM kill* — is the fingerprint of thread-per-connection hitting its wall, and you can read it off the metrics without ever seeing the code. The fix was not "add RAM" (that buys you a linear, doomed reprieve — double the RAM, survive twice the connections, die anyway); the fix was to change the *model* so memory stopped scaling with connections. We moved the I/O to an event loop and the per-connection cost dropped from a megabyte to a few kilobytes, and the same box that died at 5,000 held 50,000 without noticing. One honest caveat: the rewrite was *not* free engineering-wise — straight-line blocking handlers became async state machines, and we introduced (and then fixed) a bug where one handler did a synchronous DNS lookup that stalled the whole loop. That stall is the event loop's characteristic failure, and it is the subject of much of the rest of this track.

## When to reach for this (and when not to)

The whole point of understanding the mechanism is to make a *decision*, so let me be decisive. The choice between blocking threads and non-blocking async is not a matter of taste or fashion; it is dictated by two numbers — your concurrency $L = \lambda W$ and whether your work is I/O-bound or CPU-bound.

**Reach for non-blocking / event-loop / async when:**

- You hold **many concurrent connections that are mostly idle** — websockets, long-poll, SSE, chat, a reverse proxy, an API gateway, anything with thousands of open-but-quiet sockets. This is the canonical async case; $L$ is large and the connections are waiting, so a thread each is pure waste.
- Your work is **I/O-bound** — most of $W$ is spent waiting on the network, disk, or downstream services, not on the CPU. The waiting is what you want to make free, and async makes waiting free. (This is the same logic that, in Python specifically, sends I/O-bound work to `asyncio`; see [threading done right: io-bound concurrency and its limits](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) and [asyncio from the ground up: event loops and coroutines](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines) for the language-specific story this series deliberately does not re-derive.)
- $L = \lambda W$ comes out in the **thousands or more**. Once you need to hold thousands in flight, a thread each is untenable and the event loop is the answer.

**Stick with blocking threads (or a bounded thread pool) when:**

- Your **connection count is low** — dozens, low hundreds. A thread per connection at 200 connections is 200 MB and dead simple to write and debug. Do not async-ify a server that will never see a thousand connections; you are buying complexity you will not use. The simplest thing that works is straight-line blocking code, and for low $L$ it works fine.
- Your work is **CPU-bound** — you are crunching numbers, not waiting on I/O. Async does *nothing* for CPU-bound work; there is no waiting to overlap. CPU-bound work wants *parallelism* — a thread (or process) per core, sized to your core count — not the C10k machinery. (This distinction is the whole subject of [concurrency vs parallelism: cpu-bound, io-bound, and the scaling laws](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws).) Putting CPU-bound work on an event loop is actively *bad*: it stalls the loop and freezes every connection.
- You have **legacy blocking libraries** you cannot replace (a synchronous database driver, a blocking RPC client). A bounded thread pool lets you keep them while capping the damage — just size the pool by Little's law and add backpressure.
- You value **straight-line, debuggable control flow** over peak connection density and you are within your memory budget. Async code has real cognitive and operational costs — harder stack traces, the blocking-the-loop footgun, more complex cancellation. If you do not need it, the honest engineering call is not to pay for it.

The deciding move every time: compute $L = \lambda W$, classify the work as I/O-bound or CPU-bound, and check your memory budget against $L \times$ (1 MB per thread). If $L$ is small, threads are fine. If $L$ is large and the work is I/O-bound, you need async. If $L$ is large and the work is CPU-bound, you need *parallelism and more machines*, not async. And if you are genuinely unsure, *measure* — open connections until something breaks, watch RSS and CPU, and let the failure mode tell you which wall you are about to hit.

## Case studies / real-world

**Dan Kegel and the C10k essay (1999).** The problem this whole post is named after was crystallized by Dan Kegel in his essay "The C10k problem," which surveyed the techniques for handling ten thousand simultaneous connections and argued that the bottleneck was the *I/O model*, not the hardware. The essay catalogued blocking-thread-per-connection, `select`/`poll`, the then-emerging `epoll` (Linux) and `kqueue` (FreeBSD), and asynchronous I/O, and it became the reference that the entire event-driven-server generation grew up on. Its central claim — that you must stop dedicating a thread per connection and instead multiplex with readiness notification — is the thesis of this post, and it has aged into conventional wisdom precisely because it was right. (The essay is still online and worth reading as a historical artifact; the techniques are dated but the framing is timeless.)

**nginx versus Apache prefork.** The clearest real-world demonstration of this post's thesis is the architectural divergence between Apache's classic `prefork` MPM and nginx. Apache `prefork` used a process per connection — even heavier than a thread per connection, because a process has its own address space and is more expensive to create and switch. Under high concurrency of mostly-idle connections (the classic "slow clients" or "C10k" scenario), `prefork` Apache's memory and process-management overhead became the bottleneck; each idle keep-alive connection tied up a whole worker process. nginx, designed from the start around a small number of worker processes each running an `epoll`/`kqueue` event loop, holds tens of thousands of connections per worker with a flat memory profile. The widely-observed result — nginx serving far more concurrent connections at far lower memory than `prefork` Apache on the same hardware — is the event-loop-versus-thread/process-per-connection trade-off measured in production. (Apache later added the event-driven `event` MPM precisely to close this gap, which itself is evidence of which model wins at scale.)

**Node.js and the single-threaded event loop.** Node.js made the event-loop model mainstream for application developers by putting a single-threaded, `libuv`-backed event loop at the center of the runtime and exposing it through callbacks, then promises, then `async`/`await`. Node demonstrates both sides of the trade this post describes: it handles enormous numbers of concurrent I/O-bound connections on one thread with tiny per-connection cost (the win), and it is notoriously easy to *block the event loop* with a synchronous call or a CPU-heavy loop and freeze the whole server (the cost). Node's `libuv` thread pool — a small bounded pool that handles file I/O, DNS, and crypto behind the event loop — is itself the "use both escape routes" pattern: the loop for waiting, a bounded pool for the genuinely blocking work. The very existence of guidance like "never block the event loop" is the operational shadow of the mechanism we derived: one thread serves everyone, so one blocking call hurts everyone.

**Redis: single-threaded by design, fast because of it.** Redis is perhaps the most striking case, because it deliberately runs its command processing on a *single* thread driven by an event loop (its own small reactor over `epoll`/`kqueue`), and it is one of the fastest data stores in the world. Why does single-threaded win here? Because Redis operations are tiny and CPU-cheap, the cost is dominated by I/O multiplexing, and a single thread over an event loop has *no locking, no contention, no context switches between request handlers* — the very overheads that make a multithreaded server slower under contention simply do not exist. Redis proves the converse of the C10k lesson: when your per-request work is small and your bottleneck is handling many connections, one event-loop thread is not a compromise, it is the *optimum*, because you have eliminated synchronization entirely. (Redis did later add threads — but only for *I/O reads/writes and a few slow background tasks*, keeping the core command execution single-threaded precisely to preserve its lock-free simplicity.) The mechanism we built — one thread, many sockets via readiness, no thread-per-connection — is not just a way to *survive* C10k; in the right workload it is the way to be *fastest*.

## Key takeaways

- A **blocking read parks the whole thread**: the kernel marks it not-runnable, removes it from the run queue, and wakes it only when data arrives. The thread does no work while parked but still holds its stack and scheduler state — you pay for it to wait.
- A thread costs **roughly a megabyte** (mostly stack) plus kernel bookkeeping plus a context-switch tax of a few microseconds. That cost is fine for a handful of threads and fatal for tens of thousands.
- The **thread-per-connection model hits a memory wall, not a CPU wall**: at 10k mostly-idle connections you need ~10 GB of stacks while the cores sit idle. That is the **C10k problem** — the cost grows linearly in connections with a large constant.
- **Non-blocking sockets** (`O_NONBLOCK`) make I/O return `EAGAIN` instead of parking the thread — but using them in a naive poll loop is the **busy-poll trap**: one core pegged at 100% asking sockets that have no data.
- **Readiness notification** (`epoll`, `kqueue`) is the escape: register your sockets once, sleep until the kernel reports which are ready, and service only those — O(1) per ready event, one thread for ten thousand connections. **Completion-based** models (`io_uring`, IOCP) go further: the kernel does the I/O and tells you it is done.
- **Little's law** $L = \lambda W$ is the sizing tool: in-flight concurrency equals arrival rate times service time. **Duration ($W$), not rate ($\lambda$), is what explodes your concurrency** — I/O-bound work inflates $W$, which inflates $L$, which is why I/O-bound work wants concurrency-without-threads.
- Two escape routes: a **bounded blocking thread pool** (keeps straight-line code, caps memory, relocates pressure to a queue and backpressure) or a **single-threaded event loop** (collapses memory to kilobytes per connection but inverts control flow and must never block). Mature systems use both — the loop for waiting, the pool for working.
- **Reach for async** when you hold many idle connections and the work is I/O-bound and $L$ is in the thousands. **Stick with threads** when connection counts are low, the work is CPU-bound (that wants parallelism, not async), or you are within your memory budget and value straight-line code. When unsure, **measure** — let the failure mode tell you which wall you are hitting.

## Further reading

- Dan Kegel, "The C10k problem" — the founding essay that named the problem and surveyed `select`/`poll`/`epoll`/`kqueue`/async I/O.
- The Linux `epoll(7)` and `epoll_wait(2)` man pages, and `kqueue(2)` on BSD/macOS — the primary sources for readiness notification.
- The `io_uring` documentation and Jens Axboe's "Efficient IO with io_uring" — the modern completion-based Linux interface.
- John Little's original 1961 result and any queueing-theory text (e.g. Kleinrock) for the proof and conditions of $L = \lambda W$.
- "The Reactor pattern" (Schmidt et al., *Pattern-Oriented Software Architecture*) — the design pattern behind every event loop.
- Within this series: [why concurrency is hard and why you can't avoid it](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it) for the framing; [processes, threads, and how the OS scheduler runs them](/blog/software-development/concurrency/processes-threads-and-how-the-os-scheduler-runs-them) for thread and context-switch mechanics; [the event loop and the reactor pattern](/blog/software-development/concurrency/the-event-loop-and-the-reactor-pattern) for building the `epoll` server; [concurrency vs parallelism: cpu-bound, io-bound, and the scaling laws](/blog/software-development/concurrency/concurrency-vs-parallelism-cpu-bound-io-bound-and-the-scaling-laws) for the I/O-vs-CPU distinction; and the capstone [the concurrency playbook: choosing the right model](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model).
- For the language-specific Python story this series links out to rather than re-deriving: [threading done right: io-bound concurrency and its limits](/blog/software-development/python-performance/threading-done-right-io-bound-concurrency-and-its-limits) and [asyncio from the ground up: event loops and coroutines](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines).
