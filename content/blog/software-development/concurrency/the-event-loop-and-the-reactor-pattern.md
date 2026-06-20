---
title: "The Event Loop and the Reactor Pattern"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "How one thread serves ten thousand connections: the run-loop, readiness multiplexing from select to epoll, the reactor and proactor patterns, and the one rule that decides whether it works at all."
tags:
  [
    "concurrency",
    "parallelism",
    "event-loop",
    "reactor",
    "epoll",
    "async",
    "io-multiplexing",
    "non-blocking-io",
  ]
category: "software-development"
subcategory: "Concurrency"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/the-event-loop-and-the-reactor-pattern-1.png"
---

Picture a chat server with ten thousand connections open at once. The obvious design — the one almost everyone reaches for first — is one thread per connection. It is easy to reason about: each thread blocks on its socket, reads a message, processes it, writes a reply, loops. The blocking is fine, because while one thread sleeps on a slow client, the scheduler runs the others. The model is clean, the code is sequential, and it works beautifully up to a few hundred connections. Then it falls over. Ten thousand threads cost gigabytes of stack memory, and the scheduler spends more time switching between them than doing work. The box is mostly idle — the connections are mostly *waiting* — yet the machine is on its knees. This is the C10k problem, and it is what this whole post exists to solve.

The way out is almost paradoxical: serve all ten thousand connections from *one* thread. Not by being clever about scheduling, but by never blocking. Instead of ten thousand threads each asleep on one socket, you have one thread asleep on *all* ten thousand sockets at once, woken only when one of them actually has data ready. The thread wakes, sees the short list of sockets that became readable or writable, runs a small handler for each, and goes back to sleep. That loop — wait for ready events, dispatch a handler for each, repeat — is the **event loop**, and the design that organizes the handlers around it is the **reactor pattern**. It is how nginx serves a hundred thousand connections per worker, how Node.js and Redis run their entire universe on a single thread, and how every `async`/`await` runtime you have ever used actually executes underneath the syntax.

This is the second great move in the series' arc. We began with [mutual exclusion](/blog/software-development/concurrency/mutual-exclusion-mutexes-and-critical-sections) and locks — the *blocking* discipline, where threads take turns. Now we cross into the *non-blocking* world, where a single thread interleaves thousands of in-flight operations without ever waiting on any one of them. The hazard is different here. There is no shared mutable state across threads to race on, because there is only one thread — but there is a new, unforgiving rule that replaces the lock: **never block the event loop**. One slow handler, one accidental synchronous database call, one tight CPU loop, and every one of your ten thousand connections freezes at once. We will derive why readiness multiplexing scales the way it does (and why the old `select` does *not*), build a minimal reactor with `epoll` in C and a non-blocking server in Go, measure what one event loop actually delivers against thread-per-connection, and be honest about the one workload where the event loop is exactly the wrong tool.

![A demultiplexer blocking on ten thousand sockets returns a short ready list to a dispatcher that routes each event to an accept read or write handler before looping](/imgs/blogs/the-event-loop-and-the-reactor-pattern-1.png)

The figure above is the whole architecture in one breath. One blocking call — `epoll_wait` — sleeps on every open socket. It returns the short ready list: only the sockets that have something to do right now. A dispatcher walks that list and calls the right handler for each — an accept handler for a listening socket, a read handler for a readable client, a write handler for a socket whose send buffer has drained. Each handler runs to completion, does its small non-blocking chunk of work, and returns. Then the loop goes back to `epoll_wait`. That is the event loop, the demultiplexer, the dispatcher, and the handlers — the four parts of the reactor — and the rest of this post is what each one really is.

## The run-loop: wait for ready, dispatch, repeat

Strip the event loop down to its skeleton and it is astonishingly small. It is a `while (true)` with one blocking call at the top and a dispatch in the middle:

```c
// The event loop, in pseudocode-shaped real C structure
while (running) {
    int n = wait_for_ready(events, max_events, timeout);  // BLOCK here, only here
    for (int i = 0; i < n; i++) {
        Handler *h = events[i].handler;   // which connection became ready
        h->callback(h, events[i].what);   // run its handler to completion
    }
    run_expired_timers();                 // fire any due timeouts
}
```

Read it carefully, because every important property of the event loop is visible in those six lines. There is exactly **one** place the thread ever sleeps: `wait_for_ready`. When the loop is doing useful work — running handlers — it is *not* sleeping, and crucially it is not blocked on any single connection. The blocking is amortized across every connection at once: the thread blocks on the *set* of all sockets, not on any one socket. When the kernel has nothing for us, we sleep; the instant any connection has data, we wake with the list of which ones.

The second property: handlers run **to completion, one at a time, on the loop thread**. There is no preemption inside the loop. A handler runs from its first instruction to its `return` without the loop ever interrupting it. This is what makes event-loop code so pleasant — *within a handler, you have no concurrency to worry about*. There are no locks, no races on your connection's state, because nothing else runs while your handler runs. The flip side is the cardinal rule we will keep returning to: because the loop cannot interrupt a handler, a handler that does not return promptly holds the entire loop hostage. We will spend a whole section on that.

![A timeline of one event loop iteration blocking on epoll wait then receiving N ready fds then running each ready handler in turn then firing due timers and looping back](/imgs/blogs/the-event-loop-and-the-reactor-pattern-2.png)

The timeline above traces a single iteration. The thread blocks on the readiness call, sleeping while no connection needs it. The kernel wakes it with N ready file descriptors — say three of ten thousand. The loop runs handler one, then handler N, then fires any timers whose deadline has passed (timeouts, retries, keep-alives all live on the loop too), and loops back to block again. The shape of that iteration is the entire story: **wait once, work N times, repeat.** The cost per iteration is one syscall plus the sum of the handler runtimes. If every handler is fast, the loop spins thousands of times a second and feels instantaneous to every client.

Timers deserve a note, because they are how a single-threaded loop does "do this in 5 seconds" without a sleeping thread. The loop keeps a *timer heap* — a min-heap of pending callbacks keyed by deadline. Each iteration, before blocking, it computes the time until the nearest deadline and passes that as the *timeout* argument to `epoll_wait`. So the loop blocks on I/O, but never *past* the next timer: if no socket becomes ready, the timeout expires, `epoll_wait` returns with zero ready fds, and the loop fires the due timers. This is why the readiness call takes a timeout at all — it is the loop's clock. Real production loops also have *ordered phases* within one iteration: libuv, for instance, runs timers, then pending I/O callbacks, then the poll (the `epoll_wait`), then "check" callbacks (`setImmediate` in Node), then close callbacks, in a fixed order each turn. The phase ordering is what makes the difference between `setTimeout(fn, 0)` and `setImmediate(fn)` in Node observable — they run in different phases of the same iteration. The mental shorthand "wait once, work N times" is exactly right; the production reality is "wait once, then work through several ordered queues, then repeat."

Why does this beat thread-per-connection so decisively for I/O-bound work? Because the expensive thing in thread-per-connection is *idle capacity*. Ten thousand threads, each blocked on `read`, each carrying a full stack — typically the default is between 512 KB and 8 MB of address space reserved per thread, often 1 MB — is gigabytes of memory representing connections that are doing nothing. And every time a packet arrives, the kernel must wake a thread, which means a context switch: save the registers of whatever was running, restore the thread's registers, reload the cache and TLB working set, on the order of one to a few microseconds each. With ten thousand connections trickling traffic, you pay that switch constantly. The event loop collapses all of that. One thread, one stack, no cross-thread context switches at all — the loop runs handlers as ordinary function calls. The connection state lives in a small heap object, not a megabyte of stack. You trade ten thousand sleeping threads for one busy thread and ten thousand cheap state objects.

There is a precise way to see the win. The loop's throughput ceiling is set by how much work each iteration does. If the average handler costs $c$ microseconds of CPU and the syscall to wait costs $s$ microseconds, then one loop thread processes ready events at a rate of roughly $1/(s/N + c)$ events per microsecond when $N$ events come back per wait — the fixed syscall cost is *divided across* all the events that wait returned. The more connections are ready at once, the cheaper the per-event syscall overhead. That is the deep reason a busy event loop is efficient: under load, the one expensive syscall is shared across a fat batch of ready connections, and the loop becomes almost pure handler execution.

What does "ready" actually mean at the kernel level, since the whole loop pivots on it? Every TCP socket has two kernel-side buffers: a receive buffer where incoming bytes accumulate, and a send buffer where outgoing bytes wait to be transmitted. **Readable** means the receive buffer is non-empty (or the peer closed, which is a readable event you must handle). **Writable** means the send buffer has room — you can `write` without it failing or blocking. The readiness API is, at bottom, the kernel telling you the state of those buffers without you having to poll each one. When a packet arrives and the network stack copies its payload into a socket's receive buffer, the kernel marks that socket readable and, if you registered it with an `epoll` instance, links it onto that instance's ready list. That is the mechanism: readiness is a *push* from the network stack into a ready list, not a *pull* you perform by scanning. `select` and `poll` throw that push away and re-derive it by scanning every call; `epoll` keeps it.

This also explains why your sockets must be **non-blocking** in a reactor. Readiness is a hint, not a guarantee for the future: by the time your handler runs `read` after being told "readable," another mechanism (a second thread, a `recv` you already did) might have drained the buffer, and a *blocking* `read` would then put the whole loop to sleep — the exact catastrophe the loop exists to avoid. With `O_NONBLOCK` set, that `read` returns `EAGAIN` instead of sleeping, and your handler simply moves on. Non-blocking sockets plus a readiness loop are two halves of one design; you cannot safely have one without the other.

## select and poll: O(n) and why they don't scale

The whole design hinges on that magic `wait_for_ready` call. How does one thread sleep on ten thousand sockets and wake with just the ready ones? The first answer the world had — and the one that taught everyone the C10k lesson the hard way — was `select`, and later `poll`. Understanding *why they don't scale* is the cleanest way to understand what `epoll` had to fix.

`select` takes three bitmaps — one for read-readiness, one for write, one for errors — each with one bit per file descriptor. You set the bit for every fd you care about, call `select`, and it blocks until at least one fd is ready, then *rewrites the bitmaps in place* to show which fds are now ready. Here is the shape of it:

```c
fd_set readfds;
while (1) {
    FD_ZERO(&readfds);
    for (int i = 0; i < conn_count; i++)     // rebuild the set EVERY call
        FD_SET(conns[i].fd, &readfds);
    int maxfd = highest_fd(conns, conn_count);

    select(maxfd + 1, &readfds, NULL, NULL, NULL);  // O(n) inside the kernel

    for (int i = 0; i < conn_count; i++)     // scan ALL fds to find the ready ones
        if (FD_ISSET(conns[i].fd, &readfds))
            handle(&conns[i]);
}
```

Look at where the cost is. Every single call, you rebuild the full fd set from scratch — that is $O(n)$ in user space. You pass it to the kernel, which must *internally scan all n descriptors* to check each one's readiness — that is $O(n)$ in the kernel. When it returns, you have no list of *which* fds are ready; you only have the rewritten bitmaps, so you must scan all n again to find them — a third $O(n)$. Three linear passes over every connection, on every iteration, whether one connection is ready or all of them are. With ten thousand connections and one of them readable, you do thirty thousand units of bookkeeping to process one event.

`poll` fixes the API ugliness but not the asymptotic problem. Instead of three fixed-size bitmaps it takes an array of `struct pollfd`, each carrying an fd, the events you want, and a field the kernel fills with what happened:

```c
struct pollfd fds[MAX];
for (int i = 0; i < n; i++) { fds[i].fd = conns[i].fd; fds[i].events = POLLIN; }

while (1) {
    poll(fds, n, -1);                     // kernel STILL scans all n every call
    for (int i = 0; i < n; i++)           // and you STILL scan all n for results
        if (fds[i].revents & POLLIN)
            handle(&conns[i]);
}
```

`poll` removes `select`'s hard `FD_SETSIZE` limit (1024 on most systems — a real ceiling that bit people building bigger servers) and is a little less wasteful because the array is sized to your actual fd count. But the fundamental cost is identical: the kernel scans all n descriptors each call, and you scan all n results. Both `select` and `poll` are $O(n)$ per call, where n is the number of connections you are *watching*, not the number that are *ready*.

There is a deeper reason this asymptotic cost is unavoidable for `select`/`poll`, and naming it makes the contrast with `epoll` precise. These APIs are **stateless** across calls. The kernel keeps *nothing* about your interest set between one `poll` and the next — you hand it the entire fd array every time, it scans it, it returns, and it forgets everything. So each call must rebuild the kernel's picture of "what am I waiting on" from scratch, then walk all of it. There is no place for the kernel to *remember* that fd 5,000 became readable so it can hand it to you cheaply next time, because next time you might not even be asking about fd 5,000. The whole design assumes the interest set is small and changes constantly. For a server, the interest set is *huge and stable* — ten thousand long-lived connections — which is exactly the case `select`/`poll` handle worst. The fix is to give the kernel a place to remember the interest set, and that is what `epoll`'s persistent instance is.

#### Worked example: the cost gap between watched and ready

Make it concrete. You run a server with $n = 10{,}000$ idle-ish connections — long-lived clients that send a small message every few seconds. At any given moment, suppose 10 of them are ready. With `select`, every loop iteration costs about $3 \times 10{,}000 = 30{,}000$ units of work (rebuild, kernel scan, result scan) to discover and serve 10 events — roughly **3,000 units of overhead per useful event**. As you grow to $n = 100{,}000$ connections with the same 10 ready, the cost climbs to about 300,000 units per iteration — 30,000 units of overhead per event. The overhead grows with the *idle* population, which is exactly backwards: you wanted to pay for the connections doing work, and instead you pay for the ones doing nothing. That is the wall. The fix has to make the cost proportional to the number of *ready* connections, not the number *watched* — $O(\text{ready})$ instead of $O(n)$. That is `epoll`.

## epoll, kqueue, and IOCP: O(ready) and the triggering modes

The insight behind `epoll` (Linux), `kqueue` (BSD and macOS), and IOCP (Windows) is to stop passing the whole fd set on every call. Instead, you *register* your interest in a descriptor once, with the kernel, into a long-lived object. The kernel then maintains a ready list internally — as fds become ready, it adds them to that list — and when you ask for events, it hands you only the ready ones. You never rescan the full set, and the kernel never rescans the full set, because it already knows which ones are ready.

![A matrix comparing select poll and epoll across per-call cost fd limit and how each scales showing epoll alone is order of ready and uncapped](/imgs/blogs/the-event-loop-and-the-reactor-pattern-3.png)

The matrix above is the comparison that matters. `select` is $O(n)$ with a hard 1024-fd limit. `poll` is $O(n)$ with no hard cap. `epoll` and `kqueue` are $O(\text{ready})$ with no hard cap. That asymptotic change — from "watched" to "ready" in the cost — is the entire reason the modern event loop exists. Here is the `epoll` API in three calls:

```c
int ep = epoll_create1(0);                         // create the epoll instance once

struct epoll_event ev = { .events = EPOLLIN, .data.ptr = conn };
epoll_ctl(ep, EPOLL_CTL_ADD, conn->fd, &ev);       // register interest ONCE per fd

struct epoll_event ready[MAX_EVENTS];
while (1) {
    int n = epoll_wait(ep, ready, MAX_EVENTS, -1);  // returns ONLY ready fds, O(ready)
    for (int i = 0; i < n; i++) {
        Conn *c = ready[i].data.ptr;                // no scan: kernel gave us the list
        handle(c, ready[i].events);
    }
}
```

`epoll_create1` builds the kernel object. `epoll_ctl` with `EPOLL_CTL_ADD` registers a descriptor *once* — you do not re-pass it every loop. `epoll_wait` blocks and returns an array of exactly the ready events, length n where n is the number ready, not the number registered. You loop over just those. The `data.ptr` field is the unsung hero: it lets you stash a pointer to your own connection object on each registration, so when an event comes back, you immediately have your state without any lookup. `kqueue` is structurally the same with different names (`kqueue()`, `kevent()` to both register and wait, filters like `EVFILT_READ`), and it can watch more than sockets — files, signals, timers, process exits — through one interface. Windows IOCP we will meet shortly, because it is a different *shape* of API entirely.

The mechanism that makes `epoll_wait` genuinely $O(\text{ready})$ rather than $O(n)$ in disguise is worth seeing, because the win is not free magic — it is bookkeeping the kernel does *once per state change* instead of *once per call*. Internally, the `epoll` instance holds two structures: an interest set, typically a red-black tree keyed by fd (so `epoll_ctl` add/modify/delete is $O(\log n)$), and a **ready list**, a doubly linked list of fds that have become ready since you last reaped them. The key move is a *callback*: when you register an fd, `epoll` installs a callback on that socket's wait queue. When the network stack makes the socket readable, it runs every callback on the socket's wait queue — and `epoll`'s callback simply *appends that fd to the ready list*. So the work of "noticing this fd is ready" is charged to the event that made it ready (the packet arriving), $O(1)$, not to your `epoll_wait` call. When you call `epoll_wait`, the kernel just splices off the ready list and copies those entries to you — $O(\text{ready})$, touching nothing about the thousands of idle fds. The cost moved from *per-call-over-everything* to *per-event-that-happened*, which is exactly the asymptotic improvement, and it is why the idle population stops mattering.

One subtle consequence: because `epoll`'s state lives in a kernel object you hold by file descriptor, you can share it, pass it across `fork`, or have multiple threads wait on the same instance (with care — `EPOLLEXCLUSIVE` exists precisely to avoid the thundering herd where every thread wakes for one connection). The interest set being *persistent and kernel-resident* is the structural difference from `select`/`poll`, and almost every nice property follows from it.

There is one more thing you must choose with `epoll`, and getting it wrong is one of the most common event-loop bugs: **level-triggered versus edge-triggered**.

![A matrix contrasting level-triggered and edge-triggered epoll showing when each fires and the gotcha that edge-triggered stalls if you do not drain the socket](/imgs/blogs/the-event-loop-and-the-reactor-pattern-5.png)

**Level-triggered** is the default and the intuitive one. `epoll_wait` reports a descriptor as ready *whenever there is data to read* — every call, as long as the condition holds. If a socket has 4 KB buffered and your handler reads only 1 KB, the next `epoll_wait` reports it readable again, because there is still data. It is forgiving: you can read a little, return, and the loop will come back to you. The cost is that if you ignore a readable socket, it fires every iteration, which can spin the loop.

**Edge-triggered** (the `EPOLLET` flag) reports a descriptor as ready only on the *transition* from not-ready to ready — when new data arrives. It fires *once* per arrival. If your handler does not drain the socket completely, you will **not** be told again until *more* new data arrives — and if the peer is waiting for your reply before sending more, you deadlock that connection: the data sits in the kernel buffer, you never read it, the peer never proceeds. The rule with edge-triggered is absolute: **on every notification, read in a loop until you get `EAGAIN`** (the non-blocking signal for "no more data right now"), so you have drained everything the kernel had:

```c
// Edge-triggered read: you MUST drain to EAGAIN or you lose the wakeup
while (1) {
    ssize_t n = read(fd, buf, sizeof(buf));
    if (n > 0)      { process(buf, n); continue; }   // keep going
    if (n == 0)     { close_connection(fd); break; }  // peer closed
    if (errno == EAGAIN) break;                        // drained — now safe to return
    if (errno == EINTR) continue;                      // interrupted, retry
    perror("read"); close_connection(fd); break;
}
```

Edge-triggered is more efficient under high load — fewer wakeups, since you are told only when something *changes* — and it pairs naturally with non-blocking sockets and a drain loop. Level-triggered is easier to get right and forgiving of partial reads. nginx uses edge-triggered; many simpler servers and most beginners should start level-triggered. The gotcha in the matrix above — *edge-triggered stalls the connection if you do not drain it* — is the single most common production bug in hand-written reactors, and it is worth tattooing on the inside of your eyelids.

#### Worked example: why edge-triggered demands the drain loop

Walk the failure. A client sends 5 KB in one TCP segment. Your `epoll_wait` (edge-triggered) returns the socket readable — one notification for the arrival. Your handler calls `read` once into a 4 KB buffer, gets 4 KB, processes it, and returns, leaving 1 KB unread in the kernel. The loop goes back to `epoll_wait`. The kernel has *no new arrival* to report — the 1 KB is old data, not a transition — so it does not report this socket. Your handler is never called again for it. The 1 KB sits in the buffer forever. If those bytes were the end of an HTTP request, the request never completes, the client's browser spins, and eventually times out. The fix is the drain loop above: read until `EAGAIN`, so you pull all 5 KB on the single notification you were given. With level-triggered, the same partial read is harmless — the next `epoll_wait` re-reports the socket because 1 KB still sits there. This is the precise mechanism behind "my edge-triggered server randomly hangs on large requests," and now you can see exactly why.

## The reactor pattern: readiness and the four roles

We now have the parts; the **reactor pattern** is the name for how they fit together. It was popularized by Douglas Schmidt in the 1990s (in the POSA pattern catalog) and it has four named roles. Knowing the names makes every async framework legible, because they all implement these four:

1. **The synchronous event demultiplexer** — the blocking readiness call: `epoll_wait`, `kqueue`'s `kevent`, `select`. It waits on many sources and returns the ready set. "Demultiplex" because many input channels (sockets) collapse into one stream of ready events.
2. **The event loop / dispatcher** — the `while` loop that calls the demultiplexer and routes each ready event to the right handler. It owns the `data.ptr` mapping from descriptor to handler.
3. **Event handlers** — the per-event-type callbacks: accept a new connection, read from a readable socket, write to a writable one, handle a timer. Each is a small function that does one non-blocking step.
4. **Handles** — the OS resources being watched: the file descriptors / sockets themselves.

The defining characteristic of a reactor is the word **readiness**. The demultiplexer tells you a socket is *ready* to be read — there is data waiting in the kernel buffer — and *then your handler does the actual read*. You own the buffer, you issue the syscall, you get the bytes. The reactor inverts control: instead of you calling into a library and blocking, you register handlers and the loop calls *you* when there is work. That inversion — "don't call us, we'll call you" — is the Hollywood principle, and it is why reactor code is a web of callbacks rather than a straight line. It is also why `async`/`await` was invented: to make that callback web read like sequential code again, a story we pick up in [async/await and how coroutines actually work](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work).

The cost of that inversion is real and worth naming, because it is the reason callback-based reactor code earned the nickname "callback hell." In a blocking, thread-per-connection world, a request handler reads like a recipe: read the request, query the database, format the reply, write it — four sequential lines, the stack holds your place between them, and an exception unwinds cleanly. In a raw reactor, each of those steps that *waits* must become a separate handler registered for a separate readiness event, because the loop must be free to serve other connections during every wait. So the linear recipe shatters into a chain of callbacks, each capturing the state the next one needs, with error handling threaded manually through every link. Your local variables, which a blocking version kept on the stack, now must live in a heap-allocated connection object that survives across callbacks — because the stack is gone the moment a handler returns to the loop. This is the fundamental tax of the reactor: *the stack can no longer hold your place across a wait*, so you must explicitly materialize the "where was I" state. Coroutines and `async`/`await` are precisely the machinery that automate that materialization, giving you back the linear recipe while the compiler does the callback-shattering for you — which is why understanding the raw reactor first makes `async`/`await` legible rather than magical.

A small but important point about correctness: because the reactor is single-threaded and runs handlers to completion, **you do not need locks to protect per-connection state**. A handler for connection A cannot be interrupted by a handler for connection B; they run strictly one after another on the same thread. This is the great simplification the event loop buys you — the entire category of data races vanishes, *as long as you stay on the one thread*. The moment you offload work to another thread (we will), you are back in shared-state territory and need the discipline from the [memory model](/blog/software-development/concurrency/memory-models-sequential-consistency-and-happens-before) posts. But within the loop, single-threaded execution is your lock.

The one part of the reactor that trips up first-time implementers is the **write handler**, because writing is not always immediately possible. When you `write` to a socket and the kernel send buffer is full (a slow client, a congested link), a non-blocking `write` writes only *part* of your data and returns the count, or returns `EAGAIN` having written nothing. You cannot just loop and retry — that would spin or block. The reactor answer is symmetric to reads: buffer the unsent bytes in your connection object, register interest in `EPOLLOUT` (writable) for that socket, and return to the loop. When the kernel send buffer drains, `epoll_wait` reports the socket *writable*, your write handler fires, and you flush more of the pending buffer. When the buffer empties, you *unregister* `EPOLLOUT` (or it would fire constantly, since an empty send buffer is always writable). That add-when-pending, remove-when-drained dance for `EPOLLOUT` is the other half of correct reactor I/O, and forgetting the "remove when drained" step is a classic busy-loop bug — the loop spins at 100% CPU reporting a writable socket you have nothing to write to.

## The proactor pattern: completion instead of readiness

There is a second, subtly different architecture that does the same job from the opposite direction: the **proactor**. Where the reactor reports *readiness* ("the socket is readable — now you read it"), the proactor reports *completion* ("the read you asked for is done — here are the bytes"). The distinction sounds small and is in fact profound, because it changes who does the I/O.

![A before and after figure contrasting reactor readiness where you call read yourself against proactor completion where the kernel fills your buffer and posts the result](/imgs/blogs/the-event-loop-and-the-reactor-pattern-4.png)

In the **reactor** (left), the flow is: the demultiplexer says "fd is readable," your handler calls `read` into a buffer you own, and you process the bytes. The I/O syscall happens *in your handler*, synchronously, after you are notified. In the **proactor** (right), the flow is: you *submit* an asynchronous read up front — "read 4 KB from this socket into this buffer" — and go back to the loop. The *kernel* performs the read whenever data arrives, copies it into your buffer, and posts a *completion event* to your loop. When you pick up that event, the read is already done; the bytes are sitting in your buffer. You never call `read` yourself.

Two real systems define the proactor model. **Windows IOCP** (I/O Completion Ports) is the original: you issue overlapped reads and writes with a buffer, and completed operations queue onto a completion port that one or more threads dequeue. Windows has no `epoll`; its high-performance I/O *is* completion-based, which is why cross-platform runtimes (libuv, .NET) maintain two code paths — a reactor over `epoll`/`kqueue` on Unix, a proactor over IOCP on Windows. The second, and the reason the proactor is suddenly fashionable on Linux, is **`io_uring`**. Introduced in Linux 5.1 (2019), `io_uring` gives you two ring buffers shared between user space and kernel: a submission queue where you post operations (read, write, accept, even `openat`), and a completion queue where the kernel posts results. You batch many submissions, the kernel does the actual I/O, and you reap completions — often *without any syscall at all* on the hot path, because the rings are shared memory. It is a true proactor, and on Linux it can beat `epoll` because it removes both the per-operation syscall and the separate `read` after notification — the readiness notification and the read are fused into one submitted-then-completed operation.

Here is the conceptual shape of a proactor submission with `io_uring` (using the liburing helper API, the way most people touch it):

```c
struct io_uring ring;
io_uring_queue_init(QUEUE_DEPTH, &ring, 0);

// SUBMIT an async read; the kernel will fill buf and post a completion
struct io_uring_sqe *sqe = io_uring_get_sqe(&ring);
io_uring_prep_read(sqe, fd, buf, sizeof(buf), 0);
io_uring_sqe_set_data(sqe, conn);     // tag it so we know whose read finished
io_uring_submit(&ring);

// LATER: reap completions — the read is already done, data is in buf
struct io_uring_cqe *cqe;
io_uring_wait_cqe(&ring, &cqe);
Conn *c = io_uring_cqe_get_data(cqe);
int bytes_read = cqe->res;            // result of the read; buf is already filled
process(c, buf, bytes_read);
io_uring_cqe_seen(&ring, cqe);
```

Notice there is no `read(fd, ...)` call in the completion path — `io_uring_prep_read` *is* the read, submitted to the kernel, and the completion carries its result. That is the proactor signature: **the I/O already happened.** The trade-off is that the proactor model is harder to reason about (your buffer is owned by the kernel between submission and completion — you must not touch or free it), and `io_uring` has had a rocky security history (several sandboxes disable it). But for raw throughput on Linux, completion-based I/O is the current frontier. The honest summary: reactor (`epoll`) is the proven, portable default; proactor (`io_uring`, IOCP) wins on throughput and is where high-performance Linux servers are heading.

Why is the proactor *faster*, mechanically, and not just different? Count the syscalls per request. A reactor request is at least two trips into the kernel: one `epoll_wait` to learn the socket is readable, then one `read` to fetch the bytes (and often a third for the `write` reply). Each kernel entry costs the syscall overhead — register save, mode switch, the post-Spectre mitigations that made syscalls meaningfully more expensive on modern CPUs, easily hundreds of nanoseconds each. A proactor collapses the "learn it's ready" and "do the read" into one submitted operation, so the readiness round-trip disappears entirely. And `io_uring` goes further: because the submission and completion queues are *shared memory* between you and the kernel, you can submit a batch of operations and reap a batch of completions with as few as *zero* syscalls on the hot path (in polled mode, a kernel thread drains the submission ring on its own). The proactor wins by *removing kernel boundary crossings*, which is the dominant cost once your handlers are fast. That is also why the win shows up most on small, high-rate requests — exactly the workload where the per-request syscall overhead is a large fraction of the total.

#### Worked example: counting kernel crossings per request

Put numbers on it. A trivial echo request — read a small message, write it back — costs, under a reactor: one `epoll_wait` (amortized across the batch, call it a fraction of a syscall per request under load), one `read`, one `write`. Round it to roughly two-and-a-fraction kernel crossings per request. At, say, 300 ns per crossing, that is ~700 ns of pure syscall overhead per request before any useful work — which caps a single core somewhere around 1.4 million requests/second on overhead alone. Under `io_uring` with batched submit/reap, you can fuse the read and write into submitted operations and drain completions in batches, cutting the per-request crossings toward (and in polled mode, *to*) zero, lifting the overhead ceiling severalfold. The exact numbers depend heavily on kernel version, mitigations, and batch size — treat them as the *shape* (fewer crossings → higher ceiling), not as constants you can quote. The point stands: the proactor's advantage is a syscall-count advantage, and it grows precisely as your handlers get cheaper and your request rate climbs.

## Never block the event loop

Now the rule that governs everything. Because the loop runs handlers to completion on one thread with no preemption, **the loop can only make progress in the gaps between handler returns**. Every microsecond a handler runs is a microsecond the loop is not servicing the other 9,999 connections. If a handler runs for 200 milliseconds, the loop is frozen for 200 milliseconds — every connection, every timer, every pending write, stalls. There is no scheduler to rescue you, because *you are the scheduler*, and you are stuck inside one handler.

![A before and after figure showing a handler that blocks 200 ms freezing 9999 connections versus offloading the slow work to a thread pool so the loop keeps turning](/imgs/blogs/the-event-loop-and-the-reactor-pattern-6.png)

The figure above is the rule made vivid. On the left, a handler makes a synchronous database call that blocks for 200 ms. During those 200 ms the single loop thread is parked inside that one handler; the other 9,999 connections are frozen — their data sits unread, their replies unsent, their timers unfired. Tail latency explodes for *everyone*, because of *one* slow operation on *one* connection. On the right, the handler instead submits the slow work elsewhere and returns immediately, so the loop keeps turning and the result is delivered later. The difference between a server that holds up under load and one that face-plants is almost always this discipline.

What counts as "blocking the loop"? Three things, and they are easy to do by accident:

1. **A blocking syscall.** Any synchronous I/O that waits — a plain blocking `read`/`write`, a synchronous DNS lookup (`getaddrinfo` is notoriously blocking), a synchronous file read (file I/O is *not* readiness-pollable on Linux — `epoll` does not work on regular files, which is a classic trap), a synchronous database driver. The sockets in your loop are non-blocking, but it is dangerously easy to call something else that is not.
2. **A long CPU computation.** Parsing a 50 MB JSON payload, computing a bcrypt hash, image resizing, a regex with catastrophic backtracking, JSON.stringify on a huge object. There is no I/O wait here — the CPU is genuinely busy — but the loop is just as frozen.
3. **An unbounded synchronous loop** over a large data structure, the same as the above.

Here is the bug in JavaScript, the language where this bites hardest because *everything* runs on the one loop:

```javascript
// BUG: synchronous CPU work freezes the entire Node.js event loop
const crypto = require('crypto');
server.on('request', (req, res) => {
  // pbkdf2Sync blocks the loop for ~100 ms per call — every other
  // request, timer, and socket stalls while this runs
  const hash = crypto.pbkdf2Sync(req.password, salt, 600000, 32, 'sha256');
  res.end(hash.toString('hex'));
});
```

And the fix — use the asynchronous variant, which hands the CPU work to libuv's internal thread pool and calls you back on the loop when it is done:

```javascript
// FIX: async pbkdf2 runs on libuv's thread pool; the loop stays free
server.on('request', (req, res) => {
  crypto.pbkdf2(req.password, salt, 600000, 32, 'sha256', (err, hash) => {
    if (err) { res.statusCode = 500; return res.end(); }
    res.end(hash.toString('hex'));   // back on the loop thread, instantly
  });
});
```

The mechanism of the fix is the subject of the next section, but the principle is the rule itself: **a handler must do a bounded, small amount of work and return fast.** Everything slow — blocking I/O, heavy CPU — must happen *somewhere other than the loop thread*, with the result delivered back to the loop as just another event.

#### Worked example: how one slow handler wrecks tail latency

Quantify it. Your loop services 10,000 connections, each sending a request every second, so ~10,000 requests/second. Each fast handler takes 50 microseconds, so the loop spends about $10{,}000 \times 50\,\mu s = 500$ ms of CPU per second on handlers — half the thread, comfortable headroom, p99 latency a millisecond or two. Now one request per second triggers a code path with a synchronous 200 ms call. That one handler parks the loop for 200 ms. During those 200 ms, about $0.2 \times 10{,}000 = 2{,}000$ other requests arrive and *queue*, because the loop cannot service them. When the slow handler finally returns, the loop must burn through the 2,000-deep backlog. Every one of those 2,000 requests now has at least some fraction of 200 ms added to its latency — your p99, and even your median, jumps from milliseconds to hundreds of milliseconds, from *one* slow call per second. This is the signature of a blocked event loop in production: latency that is fine on average but periodically spikes for *all* connections at once, correlated with nothing on any single connection. The cause is never the slow connection; it is the loop it froze.

## Offloading CPU work to a thread pool

The event loop is the right place for *waiting* (I/O) and the wrong place for *computing* (CPU). The resolution is to keep the loop for I/O and hand genuine CPU work to a pool of worker threads, then deliver the result back to the loop as an event. This is the canonical hybrid, and every serious runtime ships it: libuv has a default 4-thread pool (`UV_THREADPOOL_SIZE`, tunable up to 1024) that backs file I/O, DNS, and the async crypto above; Tokio has `spawn_blocking`; Netty has a separate `EventExecutorGroup` for blocking handlers.

The pattern, language-agnostic: the handler does not do the slow work itself. It packages the work, submits it to a pool, and returns to the loop *immediately*. A worker thread runs the slow work in parallel with the loop. When it finishes, it does not touch loop state directly (that would be a cross-thread race); instead it posts a completion back onto the loop — typically by writing to a self-pipe or `eventfd` the loop is watching, or pushing onto a thread-safe queue the loop drains. The loop wakes, sees the completion, and runs the continuation on the loop thread, where touching connection state is safe again.

The mechanism that wires a worker thread back into the loop deserves a moment, because it answers an obvious question: the loop is asleep inside `epoll_wait` on a set of sockets — how does a *worker thread* wake it? The trick is the **self-pipe** (or its modern form, `eventfd`): the loop creates a pipe (or `eventfd`) and registers the *read* end with `epoll` like any socket. When a worker finishes, it pushes its result onto a mutex-protected queue and then writes a single byte to the *write* end of that pipe. That write makes the pipe's read end readable, so `epoll_wait` returns with the pipe in the ready set; the loop's handler for the pipe drains the byte, pops the finished results off the queue, and runs their continuations on the loop thread. The pipe is purely a *wakeup* channel; the real data crosses on the queue. This is exactly how libuv's `uv_async_t` is implemented internally, and it is the canonical way to deliver a cross-thread event into a single-threaded loop without a busy poll.

Here it is in Go, which makes the structure unusually clear. Go's runtime *is* an event loop over `epoll` underneath (the netpoller), but it presents blocking-looking goroutines on top, so you can spawn a goroutine for the slow work and use a channel to deliver the result without any explicit pool plumbing:

```go
func handleRequest(conn net.Conn, req Request) {
    // CPU-heavy work: run it on a separate goroutine (the Go runtime
    // schedules it on another OS thread), don't block this connection's flow.
    resultCh := make(chan []byte, 1)
    go func() {
        resultCh <- expensiveHash(req.Password) // 100 ms of CPU, off the hot path
    }()

    // Meanwhile this goroutine is parked, but the runtime's netpoller keeps
    // serving every OTHER connection on a handful of OS threads.
    hash := <-resultCh
    conn.Write(hash)
}
```

Go hides the loop, so "offloading" is just "spawn a goroutine." In a runtime where you see the loop directly — say a Rust `tokio` server — the offload is explicit, because `tokio`'s async tasks must never block the executor thread either:

```rust
async fn handle(req: Request) -> Vec<u8> {
    // spawn_blocking moves CPU-heavy work onto Tokio's dedicated blocking
    // thread pool, so the async executor thread (the event loop) stays free.
    let hash = tokio::task::spawn_blocking(move || {
        expensive_hash(&req.password)   // synchronous, CPU-bound, 100 ms
    })
    .await                              // suspend this task; the loop runs others
    .expect("blocking task panicked");
    hash
}
```

The shape is identical across languages: **the loop thread submits, a worker thread computes, the result returns to the loop as an event.** This is also where the single-threaded simplicity of the reactor ends and the concurrency discipline of the rest of this series returns. The worker thread and the loop thread share the request and result, so that handoff *must* establish a happens-before edge — the channel send/receive in Go, the `.await` on the join handle in Rust, the queue with proper memory ordering in C — or you have reintroduced a data race across the boundary. The pool buys you parallelism for CPU work; the handoff is where you pay the synchronization tax.

A sizing note, because it is a real decision: the pool should be sized to *cores* for CPU-bound work (more threads than cores just adds context-switch overhead with no throughput gain — the cores are already saturated), and can be larger for *blocking-I/O* work (where threads are mostly asleep, so you want enough to cover the concurrency, like a connection-pool sized to in-flight queries). Conflating these — using one giant pool for both — is a common misconfiguration that either starves CPU work or wastes memory on idle threads.

## How real runtimes implement this

None of this is theoretical — it is the literal implementation of every async runtime you use. Strip the syntax away and you find the same event loop over the same readiness API.

![A stack diagram showing your handlers over the event loop over the runtime glue over the readiness API over the kernel sockets across ten thousand descriptors](/imgs/blogs/the-event-loop-and-the-reactor-pattern-7.png)

The stack above is universal. Your handlers (or `async` functions) sit on top. Below them, the event loop waits and dispatches. Below that, runtime glue (libuv, Tokio, Netty) adapts the loop to a platform readiness API — `epoll` on Linux, `kqueue` on BSD/macOS, IOCP on Windows. At the bottom, the kernel and the actual sockets. Every runtime is a different top layer over the same foundation.

![A matrix of four real event loops Node libuv Python asyncio Netty and Tokio mapping each language to the epoll kqueue or IOCP mechanism it wraps](/imgs/blogs/the-event-loop-and-the-reactor-pattern-8.png)

To make the "every runtime is this reactor" claim concrete in a fourth language, here is the same loop in Java with NIO's `Selector` — the exact machinery Netty wraps. `Selector` is Java's portable face over `epoll`/`kqueue`/IOCP, and the structure is the reactor, role for role:

```java
Selector selector = Selector.open();                 // the demultiplexer (epoll)
serverChannel.configureBlocking(false);
serverChannel.register(selector, SelectionKey.OP_ACCEPT);  // register interest once

while (running) {
    selector.select();                               // BLOCK once, on everything
    Iterator<SelectionKey> it = selector.selectedKeys().iterator();
    while (it.hasNext()) {
        SelectionKey key = it.next();
        it.remove();
        if (key.isAcceptable()) {                    // accept handler
            SocketChannel client = serverChannel.accept();
            client.configureBlocking(false);
            client.register(selector, SelectionKey.OP_READ);
        } else if (key.isReadable()) {               // read handler
            SocketChannel ch = (SocketChannel) key.channel();
            ByteBuffer buf = ByteBuffer.allocate(4096);
            int n = ch.read(buf);
            if (n < 0) { ch.close(); }
            else { buf.flip(); ch.write(buf); }       // echo (real code buffers writes)
        }
    }
}
```

`selector.select()` is the demultiplexer, the `while` over `selectedKeys()` is the dispatcher, and `isAcceptable`/`isReadable` route to handlers — the identical four roles as the C version, in idiomatic Java. Netty's entire value-add is wrapping this in a clean handler pipeline, pooled `ByteBuf`s, and a worker `EventLoopGroup`, so you never write this loop by hand. Now the matrix.

The matrix names four of them. **Node.js / libuv** runs a single-threaded loop in C (libuv) under the V8 JavaScript engine. libuv abstracts `epoll`/`kqueue`/IOCP and adds the thread pool for file I/O and DNS; the JavaScript you write is callbacks and promises resolved on that one loop. **Python `asyncio`** runs an event loop driven by the `selectors` module, which picks `epoll`/`kqueue` for you, scheduling coroutines as `await`-able tasks — though the GIL means it is genuinely single-threaded for Python bytecode, which is precisely why `asyncio` shines for I/O and not CPU. The full Python-specific story — the loop, coroutines, the GIL's role — is told in [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines); this post is the language-agnostic skeleton beneath it. **Netty** (the JVM networking framework behind gRPC-Java, Cassandra's transport, and much of the Java async world) runs `EventLoopGroup`s, each an event loop over Java NIO's `Selector` (which is `epoll` internally on Linux), with a `ByteBuf` allocator and a pipeline of handlers. **Tokio** (Rust) runs a multi-threaded work-stealing scheduler over `mio` (which wraps `epoll`/`kqueue`/IOCP), where `async fn`s compile to state machines that the executor polls — the reactor and the futures are first-class, not hidden.

The crucial unifying insight: `async`/`await` in every one of these is *not* a different concurrency model from the event loop — it is **syntax sugar over the event loop**. An `await` point is exactly where a handler returns control to the loop so the loop can run other work; when the awaited operation's event arrives, the loop resumes your function from where it paused. The callbacks of a raw reactor and the `await`s of an async function are the *same control flow*, written two ways. That is why blocking inside an `async fn` is just as fatal as blocking in a callback: both freeze the same loop. Knowing the reactor is underneath turns `async`/`await` from magic incantation into a tool you can reason about and debug.

#### Worked example: tracing an `await` to the loop

Trace one request through Tokio to see the loop underneath the syntax. Your `async fn handle` runs on a loop thread until it hits `let n = socket.read(&mut buf).await`. The `.await` calls the future's `poll`; the socket has no data yet, so `poll` returns `Pending` after registering the socket with `mio` (i.e., `epoll_ctl ADD` for readability) and stashing a *waker*. Your function *returns* — the loop thread is now free and runs other tasks. Later, `epoll_wait` reports the socket readable; the reactor finds the registered waker and calls it, which reschedules your task onto the run queue. A loop thread picks it up, polls your future again, `read` now succeeds, returns `Ready(n)`, and your function resumes right after the `.await` with `n` bytes. That entire dance — suspend at `.await`, register with `epoll`, return to the loop, get woken on readiness, resume — is the reactor pattern wearing an `async` costume. The `.await` is the handler return; the waker is the dispatch.

## A minimal reactor, end to end

Theory lands when you can build it. Here is a complete minimal reactor in C using `epoll` — an echo server that accepts connections, reads, and echoes back, all on one thread, level-triggered for clarity. This is the whole pattern in under sixty lines.

```c
// A complete minimal reactor: epoll-based echo server, one thread, level-triggered
#include <sys/epoll.h>
#include <sys/socket.h>
#include <fcntl.h>
#include <unistd.h>
#include <string.h>

static void set_nonblocking(int fd) {
    int flags = fcntl(fd, F_GETFL, 0);
    fcntl(fd, F_SETFL, flags | O_NONBLOCK);   // sockets MUST be non-blocking
}

int run_reactor(int listen_fd) {
    set_nonblocking(listen_fd);
    int ep = epoll_create1(0);

    // Register the listening socket so we get told about new connections.
    struct epoll_event ev = { .events = EPOLLIN, .data.fd = listen_fd };
    epoll_ctl(ep, EPOLL_CTL_ADD, listen_fd, &ev);

    struct epoll_event ready[1024];
    while (1) {
        int n = epoll_wait(ep, ready, 1024, -1);   // BLOCK once, on everything
        for (int i = 0; i < n; i++) {
            int fd = ready[i].data.fd;

            if (fd == listen_fd) {
                // ACCEPT handler: drain the accept queue (level-triggered ok with one)
                int client = accept(listen_fd, NULL, NULL);
                if (client < 0) continue;
                set_nonblocking(client);
                struct epoll_event cev = { .events = EPOLLIN, .data.fd = client };
                epoll_ctl(ep, EPOLL_CTL_ADD, client, &cev);  // watch the new client
            } else {
                // READ handler: read available bytes and echo them back
                char buf[4096];
                ssize_t r = read(fd, buf, sizeof(buf));
                if (r > 0) {
                    write(fd, buf, r);              // echo (real code buffers writes)
                } else if (r == 0 || (r < 0 && errno != EAGAIN)) {
                    epoll_ctl(ep, EPOLL_CTL_DEL, fd, NULL);  // unregister
                    close(fd);                                // and close
                }
            }
        }
    }
}
```

Map it back to the four roles. `epoll_wait` is the **demultiplexer**. The `while` loop and the `if (fd == listen_fd)` branch is the **dispatcher**. The accept branch and the read branch are the **handlers**. The `data.fd` field is how each ready **handle** identifies itself. There are no threads, no locks; one thread serves every connection. (Production reactors do more: they buffer partial writes for when the send buffer is full, watch `EPOLLOUT` to know when a slow socket can accept more, handle `EPOLLHUP`/`EPOLLERR`, and use edge-triggered with drain loops — but the skeleton is exactly this.)

The same shape in Go reads completely differently, because the Go runtime hides the loop. You write blocking-looking code and spawn a goroutine per connection; the runtime's netpoller (an `epoll` reactor) multiplexes them all onto a few OS threads underneath:

```go
func serve(ln net.Listener) {
    for {
        conn, err := ln.Accept()      // looks blocking; runtime parks on epoll
        if err != nil { return }
        go handleConn(conn)           // one goroutine per connection — cheap
    }
}

func handleConn(conn net.Conn) {
    defer conn.Close()
    buf := make([]byte, 4096)
    for {
        n, err := conn.Read(buf)      // looks blocking; goroutine suspends on epoll
        if err != nil { return }      // until the netpoller says this fd is readable
        conn.Write(buf[:n])           // echo
    }
}
```

This is the great trick Go and the `async` languages pull: you write sequential, blocking-*looking* code, and the runtime turns it into reactor callbacks internally. The `conn.Read` that *looks* like it blocks the thread actually suspends just the goroutine and parks it on the netpoller's `epoll`; the OS thread goes off to run other goroutines. A goroutine's stack starts at ~2 KB (versus a megabyte for an OS thread) and grows on demand, which is why a million goroutines is routine where a million OS threads is impossible. Same reactor, same `epoll`, two radically different programming experiences — one where you see the loop (C), one where the runtime hides it (Go, and the `async` languages).

## Measured behavior: one loop versus thread-per-connection

The claim of this whole post is that one event-loop thread beats thousands of blocking threads for I/O-bound load. Here is how that actually measures out, and how to measure it honestly. These are representative order-of-magnitude figures consistent with the well-documented benchmarks behind the C10k literature, nginx vs Apache-prefork comparisons, and libuv/Netty load tests — they are *approximate* and vary with kernel, hardware, and workload; treat them as the shape of the result, not exact constants.

| Metric (10,000 idle-ish conns) | Thread-per-connection | Single event loop |
| --- | --- | --- |
| Memory for connection state | ~10 GB (1 MB stack each) | ~tens of MB (small heap objects) |
| Threads / context switches | 10,000 threads, constant switching | 1 thread, ~zero cross-thread switches |
| Cost to wait on all conns | implicit per-thread block | one `epoll_wait`, O(ready) |
| Throughput ceiling | scheduler-bound, degrades past ~1k | CPU-of-handlers-bound, flat to 100k+ |
| p99 latency under light load | low until thread count bites | low and stable |

The shape is unambiguous: thread-per-connection is *fine* up to hundreds or low thousands of connections, then memory and scheduler overhead bend it down; the event loop stays flat into the hundreds of thousands because it adds essentially nothing per idle connection. The crossover is somewhere in the low thousands of connections on typical hardware — below it, thread-per-connection's simplicity often wins; above it, the event loop is the only thing that survives.

But there is a second table that is just as important, because it is the *failure mode* of the event loop — and it is the one benchmark people forget to run:

| Workload | Event loop p99 | Thread pool p99 |
| --- | --- | --- |
| 10k conns, all fast (50 µs) handlers | low, stable (~ms) | similar, more memory |
| 10k conns, 1/sec triggers 200 ms sync CPU | catastrophic spikes (hundreds of ms for all) | absorbed by other threads |
| Pure CPU-bound (no I/O wait) | no benefit; one core only | scales with cores |

The first row is the event loop's home turf and it wins. The second row is the trap from the "never block" section: a single slow synchronous call per second wrecks p99 for *every* connection on a pure event loop, while a thread pool absorbs it because other threads keep serving. The third row is the honest limit: an event loop is *one thread*, so for genuinely CPU-bound work it uses *one core* and a thread pool (or multiple loop threads) wins outright. This is why production servers run *multiple* event loops — one per core (nginx workers, Node cluster, Netty's `EventLoopGroup`, Tokio's multi-threaded runtime) — to get both the per-loop efficiency and multi-core parallelism.

How to measure this honestly: warm up first (JIT, page faults, connection establishment all distort the first runs); use a real load generator (`wrk`, `wrk2`, `h2load`) with a fixed request rate, *not* a closed-loop "as fast as possible" client, because closed-loop hides queueing latency (this is the coordinated-omission trap Gil Tene documented — measure latency at a *fixed open-loop rate* or your p99 is a fantasy); report percentiles, never just the mean (the mean hides exactly the tail spikes that the event loop's blocking failure produces); run long enough to see periodic stalls; and name your platform, because `epoll` vs `io_uring`, kernel version, and core count all move the numbers. A benchmark that reports "120k requests/second" with no latency distribution and no rate control is telling you almost nothing.

There is one event-loop-specific metric you should monitor in production, not just in benchmarks: **event-loop lag**. The idea is to schedule a timer for, say, every 100 ms and measure how *late* it actually fires. On a healthy loop, a 100 ms timer fires at ~100 ms; the lag is sub-millisecond. When a handler blocks the loop for 200 ms, that timer cannot fire until the loop is free again, so it fires ~100 ms late, and the lag metric spikes to ~200 ms. Loop lag is therefore a direct, cheap readout of "is any handler blocking me," and it is the single most useful health signal for an event-loop server — Node exposes it via `perf_hooks.monitorEventLoopDelay`, and most production Node/Tokio deployments alert on it. If your p99 latency is spiking and your loop-lag is spiking in lockstep, you have a blocked-loop problem; if latency spikes while loop-lag stays flat, the bottleneck is downstream, not in your loop. That single distinction will save you hours of misdirected debugging.

## Case studies / real-world

**nginx** is the canonical proof. Built specifically to beat the thread/process-per-connection Apache prefork model at C10k, nginx runs a small fixed number of single-threaded worker processes (typically one per CPU core), each an `epoll`-based event loop (edge-triggered, with drain loops) serving many thousands of connections. The architecture is exactly the reactor in this post, multiplied by core count. It is why a single modest nginx box routinely fronts a hundred thousand-plus concurrent connections on memory that thread-per-connection could not dream of — the per-connection cost is a small state struct, not a thread. The C10k problem essay by Dan Kegel (1999, updated through the 2000s) is the document that crystallized the whole shift, and nginx is its most successful answer.

**Redis** runs its core command processing on a *single thread* with an event loop (its own `ae` library over `epoll`/`kqueue`), and this is a deliberate design choice, not a limitation. Because commands execute one at a time on one thread, every Redis command is atomic *for free* — no locks, no races on the keyspace, which is the same single-threaded simplification the reactor buys. Redis handles enormous throughput (hundreds of thousands of ops/second) precisely because in-memory operations are microseconds-fast, so the loop never stalls. Its one rule is the rule of this post: a slow command (a big `KEYS *`, an O(n) operation on a huge collection) *blocks the whole server*, because it blocks the one loop — which is exactly why Redis documentation warns against those commands. (Redis 6+ added I/O threads for network read/write, but command execution stays single-threaded, keeping the atomicity guarantee.)

**Node.js / libuv** brought the reactor to the mainstream and, with it, the mainstream's most common production incident: the blocked event loop. Real outages have traced to a synchronous `JSON.parse` of a large payload, a synchronous `fs.readFileSync` in a request path, a regular expression with catastrophic backtracking (ReDoS), or a synchronous crypto call — each freezing the single loop and spiking latency for every concurrent request, exactly the worked-example failure above. The libuv thread pool (default size 4) exists precisely to keep file I/O, DNS, and async crypto *off* the loop. The lesson the Node ecosystem learned the hard way, and now enforces with tooling like the event-loop-lag metric, is the cardinal rule stated plainly: **the moment any handler blocks, the whole server blocks.**

## When to reach for this (and when not to)

The event loop is the right tool with a sharp boundary. Reach for it when:

- **The work is I/O-bound with high concurrency.** Many connections, each mostly waiting on the network, disk, or another service — web servers, proxies, API gateways, chat, real-time push, databases' network frontends. This is the home turf, and it is decisive above a few thousand connections.
- **Per-operation work is small and bounded.** Each handler does a quick, predictable amount of work and returns. If you can guarantee every handler is fast, the loop stays responsive.
- **You want single-threaded simplicity for shared state.** One loop thread means no locks on per-connection or per-keyspace state — Redis's atomicity, Node's programming model. That simplification is worth real money in correctness.

Do **not** reach for it when:

- **The work is CPU-bound.** An event loop is one thread, so it uses one core. For computation-heavy work — number crunching, ML inference, video transcoding, large parsing — a thread pool (or multiple processes/loops) sized to your cores is the right tool, and a single loop is strictly worse. Don't put a CPU job on the loop; offload it or don't use a loop at all.
- **You cannot guarantee handlers are fast.** If your workload contains unavoidable blocking calls (a legacy synchronous driver, a library with no async API, blocking file I/O) and you cannot offload them, a thread-per-request or bounded-thread-pool model is safer — it degrades gracefully where the loop face-plants. Don't fight a fundamentally blocking dependency by wrapping it in async theatre; either offload it to a real thread pool or use threads.
- **Concurrency is low.** With a few dozen connections, thread-per-connection is simpler, easier to debug (a real stack trace per request, no callback soup), and costs nothing. Don't add a reactor's complexity to a problem that ten threads solve. The event loop earns its keep at *scale*; below it, prefer the simpler thing.

The decisive test is the one from the [intro to this series](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it): is the work *waiting* or *computing*? Waiting at high concurrency → event loop. Computing → threads/cores. Mixed → an event loop for the I/O with a thread pool for the compute, which is what every grown-up runtime ships. The full decision framework across every model lives in the [concurrency playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model); the event loop is the answer to one specific, very common question, not a universal hammer. And once you understand the loop, the natural next question — how `await` makes its callbacks read like sequential code — is exactly [where coroutines come in](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work).

## Key takeaways

1. **The event loop is a run-loop: wait for ready events, dispatch a handler for each, repeat.** One thread blocks on *all* connections at once and wakes only for the ones with work, turning ten thousand sleeping threads into one busy thread plus ten thousand cheap state objects.
2. **`select` and `poll` are O(n) in connections watched; `epoll`/`kqueue`/IOCP are O(ready).** The old APIs rescan every descriptor every call; the modern ones register interest once and return only the ready set. That asymptotic shift — from watched to ready — is why C10k became solvable.
3. **The reactor reports readiness, the proactor reports completion.** With a reactor (`epoll`) you are told a socket is readable and *you* do the read; with a proactor (`io_uring`, IOCP) you submit the read and the kernel hands you finished bytes. Proactor wins on throughput; reactor wins on portability and is the proven default.
4. **Edge-triggered fires once on the transition — drain to `EAGAIN` or you lose the wakeup.** Level-triggered re-fires while data remains and is forgiving; edge-triggered is efficient but stalls the connection if your handler does not read everything. This is the most common hand-written-reactor bug.
5. **Never block the event loop — it is the one unforgiving rule.** Because handlers run to completion on one thread with no preemption, a single blocking syscall or heavy CPU loop freezes *every* connection at once and spikes tail latency for all of them. The slow connection isn't the victim; the loop it froze is.
6. **Offload blocking and CPU work to a thread pool, then return the result to the loop as an event.** Keep the loop for waiting, hand computing to workers sized to your cores, and synchronize the handoff (channel, join, queue) — that handoff is where the rest of this series' concurrency discipline returns.
7. **`async`/`await` is syntax sugar over the event loop, not a separate model.** An `await` is exactly a handler return; the waker is the dispatch. libuv, asyncio, Netty, and Tokio are all the same reactor over `epoll`/`kqueue`/IOCP under different language idioms.
8. **One loop is one core — run one loop per core for multi-core scale.** nginx workers, Node cluster, Netty `EventLoopGroup`, and Tokio's multi-threaded runtime all multiply the single-loop efficiency across cores. For CPU-bound work, a single loop is the wrong tool.

## Further reading

- Dan Kegel, **"The C10k Problem"** — the 1999 essay that named the problem and catalogued the I/O models (`select`, `poll`, `epoll`, async I/O), the historical anchor for everything here.
- **`epoll(7)`, `kqueue(2)`, and `io_uring` man pages and the io_uring "What's new" papers (Jens Axboe)** — the primary sources for the readiness and completion APIs, including the edge-triggered semantics.
- Douglas C. Schmidt et al., **"Pattern-Oriented Software Architecture, Volume 2" (the Reactor and Proactor patterns)** — the formal pattern definitions and the demultiplexer/dispatcher/handler vocabulary.
- **The libuv design overview** and **the Node.js "Don't Block the Event Loop" guide** — how a production loop and its thread pool are actually built, and the canonical statement of the cardinal rule.
- **The Tokio tutorial** (the runtime, the reactor, `spawn_blocking`) and **Netty's "EventLoop and threading model" docs** — two production reactors with the loop made first-class.
- Gil Tene, **"How NOT to Measure Latency"** — why open-loop, percentile, coordinated-omission-aware measurement is the only honest way to benchmark a server, and the source of the measurement discipline above.
- Within this series: the [intro](/blog/software-development/concurrency/why-concurrency-is-hard-and-why-you-cant-avoid-it), the sibling on [blocking vs non-blocking I/O and the C10k problem](/blog/software-development/concurrency/blocking-vs-non-blocking-io-and-the-c10k-problem), the forward link to [async/await and coroutines](/blog/software-development/concurrency/async-await-and-how-coroutines-actually-work), and the [capstone playbook](/blog/software-development/concurrency/the-concurrency-playbook-choosing-the-right-model). For the Python-specific loop, [asyncio from the ground up](/blog/software-development/python-performance/asyncio-from-the-ground-up-event-loops-and-coroutines).
