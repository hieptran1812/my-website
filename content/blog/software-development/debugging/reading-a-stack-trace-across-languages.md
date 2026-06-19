---
title: "Reading a Stack Trace Across Languages: Where It Crashed Is Not Where the Bug Is"
date: "2026-06-19"
publishDate: "2026-06-19"
description: "Learn to read a stack trace the way a senior engineer does in Python, Java, Go, Node, and C++, follow the Caused-by chain to the real root, and recover the frames that async and optimization throw away."
tags:
  [
    "debugging",
    "software-engineering",
    "stack-trace",
    "exception-handling",
    "symbolization",
    "gdb",
    "async",
    "root-cause-analysis",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/reading-a-stack-trace-across-languages-1.png"
---

It is 3am and the pager goes off. The on-call dashboard shows a spike of 500s, and the error tracker has helpfully grouped ten thousand of them under one headline: `NullPointerException`. You click in. There is a stack trace forty lines long. You read the first line. It points at a logging filter inside the web framework — code you have never touched and never will. You sigh, because the first line is almost never your code, and you have just learned nothing. The engineer who got paged before you read that same first line, decided "the framework is broken," restarted the pods, and went back to bed. The 500s came back twelve minutes later, which is why you are awake now.

Here is the thing almost nobody is taught explicitly: **the stack trace is the single most-read artifact in all of debugging, and most engineers read it wrong.** They read the top line and stop. Or they read the bottom line and stop. Or they skim the whole thing looking for a word they recognize and grab the first one. A stack trace is not a sentence you read once; it is a *map of a journey*, printed at the moment the journey crashed, and there is a craft to reading it — a craft that changes from language to language, because Python, Java, Go, Node, and C++ each print the map in a different order and bury the real cause in a different place. The line that crashed is rarely the line with the bug. The frame at the top is rarely the frame you should fix. And the real cause is very often four levels down a chain of "Caused by:" that you stopped reading after the first one.

![A vertical stack of call frames showing the innermost crash frame on top, library and user frames below it, and a note that the real bug lives two frames up from the crash](/imgs/blogs/reading-a-stack-trace-across-languages-1.png)

The figure above is the whole idea in one picture: a call stack is a last-in-first-out pile of frames, the crash happens at the top, and the bug is usually two frames down in your code, not in the library helper that happened to dereference the bad value. By the end of this post you will be able to do six concrete things. You will read a stack trace fluently in Python, Java/JVM, Go, Node/JavaScript, and C/C++, knowing for each one which end to start from. You will follow a "Caused by:" chain — or Python's `raise ... from`, or the "During handling of the above exception" message — down to the actual origin instead of fixing the symptom on top. You will recognize the frames that *lie*: the async stacks that drop the logical caller, the inlined frames that `-O2` merged, the truncated "... 23 more", the framework noise you must filter out, and the trace that came from a different thread than the one that failed. You will get a *good* trace in the first place — symbolizing a stripped binary with `addr2line`, turning on `faulthandler` in Python, building with `-g`, enabling Node's async stack traces, and pulling a full goroutine dump out of a hung Go process. And you will turn the frame sequence into a hypothesis by reading the argument values in the frames with `gdb` and `pdb`. This is the *observe* stage of the series' master loop — observe, reproduce, hypothesize, bisect, fix, prevent — and reading the trace correctly is how you observe what the program was actually doing when it died. If you have not read it, [the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) sets up that loop; this post is about reading the very first piece of evidence the crash hands you.

## 1. What a stack trace actually is

Before we read traces in five languages, we have to be exact about what a stack trace *is*, because the confusion that makes people misread it is almost always a confusion about the underlying machine. A program's threads each have a **call stack**: a region of memory that grows and shrinks as functions call other functions. Every time a function is called, the runtime pushes an **activation record** — also called a **stack frame** — onto the stack. That frame holds the function's local variables, its arguments, the return address (where to jump back to when this function finishes), and some bookkeeping. When the function returns, its frame is popped off. The stack is therefore a textbook **LIFO** structure — last in, first out — and at any instant it records the exact chain of calls that got you to where you are: `main` called `handle_request`, which called `parse_body`, which called `json_decode`, which called `read_byte`, and right now we are inside `read_byte`. Five frames, five activation records, stacked on top of each other.

A **stack trace** is simply a snapshot of that stack, printed as text, taken at the moment something went wrong. When an exception is thrown (or a panic, or a fatal signal), the runtime walks the stack from the current frame outward, and for each frame it records the function name, the file, and the line number of the call site. That walk is called **unwinding**: the exception propagates up the stack frame by frame, and if no frame catches it, the runtime prints the whole walk and the program dies. So a stack trace is literally a list of the activation records that were live when the program broke, in the order the unwinder visited them.

This gives us the two most important vocabulary words in the whole post. The **innermost frame** is the one closest to where the crash happened — the function that was actually executing when the exception fired, like `read_byte` in the example. The **outermost frame** is the one furthest from the crash, the entry point, usually `main` or a thread's run loop. Every language prints both, but — and this is the source of more misreading than anything else — *they disagree about which one to print first.* Some print innermost-first (the crash is at the top, you read down toward `main`). Some print innermost-last (the crash is at the bottom, you read up toward it). If you do not know which convention you are looking at, you will start at the wrong end and reason backward.

And here is the deeper truth, the one I want you to carry into every trace you ever read: **"where it crashed" and "where the bug is" are different questions.** The innermost frame tells you where the program *detected* a problem — where it tried to dereference a null pointer, index past the end of an array, divide by zero, or open a file that was not there. But the *bug* — the actual mistake a human made — is usually somewhere up the stack, in the code that passed the bad value down. The null pointer was dereferenced in a library's `String.length()` call, sure, but the *bug* is three frames up where your code forgot to handle the case where the user has no display name and passed `null` into the formatter. The library did exactly what it was told. The trace shows you the crash site for free; finding the bug site is the reading skill. We will come back to this idea in every section, because it is the single most valuable habit in this entire post.

One more piece of mechanism, because it explains a class of confusing traces. The stack you get printed is the stack of the thread that *threw*. If your program has many threads, and thread A computed a bad value and handed it to thread B through a queue, and thread B crashed on it, the trace you get is thread B's stack — which contains no mention of thread A at all. The frame that created the bad value is not in the trace, because it is on a completely different stack that already unwound. This is why concurrency bugs produce traces that "make no sense": the crash frame is innocent, the frames above it are innocent, and the guilty party is on a stack you were never shown. Keep that in your pocket; it explains a lot of 3am confusion, and it is the bridge to the sibling post on [debugging async and event loops](/blog/software-development/debugging/debugging-async-and-event-loops).

### How unwinding actually works, and why it can lie

It is worth being precise about the *unwinding* mechanism, because almost every "lying" trace later in this post is a consequence of how unwinding is implemented. When an exception is thrown, the runtime needs to do two things: find a handler (a `catch`/`except`/`recover` willing to take this exception type) and, on the way to it, run every cleanup action between here and there (destructors in C++, `finally` blocks in Java/Python, `defer` in Go). To do that it walks the stack outward. But *how* it knows the layout of each frame — where the return address is, where the locals live, which cleanup to run — is not free information. In C++ and Rust the compiler emits **unwind tables** (the DWARF `.eh_frame` section) that describe, for every instruction address, how to find the caller's frame and what to clean up. The unwinder reads those tables to walk the stack. This is the crucial point: **the trace is only as accurate as the metadata the compiler emitted**, and at high optimization levels that metadata gets thinner — a frame that was inlined has no separate unwind entry, a tail call left no frame to describe, a variable kept only in a reused register is marked "no location here." The frames you see are a *reconstruction* from tables, not a direct readout of some authoritative list, and reconstructions can be incomplete.

Managed runtimes do it differently but hit the same wall. The JVM and CPython maintain the call chain in data structures the runtime owns (the JVM's per-thread frame stack, CPython's chain of frame objects linked by `f_back`), so they can always produce an accurate trace of *managed* frames cheaply — which is why a Python or Java traceback never has `?? ()` frames the way a stripped C binary does. The cost shows up elsewhere: the moment execution crosses into native code (a C extension, a JNI call, a syscall), the managed runtime loses visibility, and the trace either stops at the boundary or needs a separate native unwinder to continue. So the boundary between "managed frames the runtime tracks" and "native frames you must symbolize" is exactly where Python's `faulthandler` and the JVM's native-frame handling earn their keep. Understanding that there are *two unwinders* — the managed one that is always accurate and the native one that depends on tables and symbols — explains why a Python segfault inside NumPy needs `faulthandler` (to print the Python half) *and* a symbolized core dump (to print the C half): no single unwinder spans the whole stack.

There is also a real, measurable *cost* to capturing a trace, and it shapes how runtimes behave under load. Capturing a stack trace means walking N frames and, for each, resolving a function name and line — for a deep stack that is not nothing, and in a hot path that throws exceptions for control flow (an anti-pattern, but common) it shows up in a profiler. This is why the JVM has an optimization called **fast-throw**: after the same exception is thrown at the same spot enough times, HotSpot stops filling in the stack trace and throws a pre-allocated exception with *no trace at all*. You see a bare `java.lang.NullPointerException` with zero frames and think the logging is broken. It is not — the JVM elided the trace for speed. The fix is the flag `-XX:-OmitStackTraceInFastThrow`, which forces full traces back on. That a runtime will *delete your evidence to go faster* is the kind of thing you only learn at 3am, and it is a direct consequence of trace capture costing real cycles.

## 2. Python: read it bottom-up

Let us start with Python, because Python's convention is the one that confuses people coming from Java and vice versa, and because Python actually prints a helpful instruction at the top that everyone ignores. Here is a real traceback from a small program that crashes parsing a config:

```python
Traceback (most recent call last):
  File "app.py", line 41, in <module>
    main()
  File "app.py", line 34, in main
    cfg = load_config("settings.yaml")
  File "config.py", line 18, in load_config
    return parse_timeout(raw["timeout"])
  File "config.py", line 9, in parse_timeout
    return int(value) * 1000
ValueError: invalid literal for int() with base 10: '30s'
```

Read the very first line: **"Traceback (most recent call last)"**. That sentence is the instruction manual, and it tells you the printing order. The *most recent call* — the innermost frame, the one closest to the crash — is printed **last**, at the bottom. So in Python you read **bottom-up**. The bottom line, `ValueError: invalid literal for int() with base 10: '30s'`, is the exception type and message: an `int()` call got the string `'30s'`, which is not a valid integer. The frame directly above it, `parse_timeout` at `config.py:9`, is where the crash happened — `int(value) * 1000` blew up because `value` was `'30s'`. That is the *crash site*.

But where is the *bug*? Read up one more frame. `load_config` at `config.py:18` called `parse_timeout(raw["timeout"])`, passing in `raw["timeout"]` — which came straight from the YAML file. So the program is reading a timeout of `'30s'` from a config file and trying to `int()` it directly, with no handling for human-friendly suffixes like `s` for seconds. The *bug* is the assumption, made in `load_config` and visible at line 18, that the timeout field is always a bare integer. The crash is at line 9; the bug is the contract violation that line 18 should have caught. You found that by walking *up* from the crash, frame by frame, asking at each one "did this frame pass something wrong to the frame below it?"

Now, the crucial complication: **chained exceptions**. Real Python code wraps and re-raises, and when it does, you get two tracebacks glued together. There are two glue messages, and they mean very different things:

```python
Traceback (most recent call last):
  File "db.py", line 22, in connect
    sock = socket.create_connection((host, port), timeout=2)
ConnectionRefusedError: [Errno 111] Connection refused

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "app.py", line 12, in <module>
    start_server()
  File "app.py", line 8, in start_server
    db = connect_with_retry()
  File "db.py", line 31, in connect_with_retry
    raise DatabaseError("could not reach database") from exc
DatabaseError: could not reach database
```

The message **"The above exception was the direct cause of the following exception"** means the code used `raise NewError(...) from exc` — explicit exception chaining. The exception printed *first* (the `ConnectionRefusedError` at the top) is the **root cause**, and the exception printed *last* (the `DatabaseError`) is the high-level wrapper. So with chaining, the reading order flips your intuition: the *first* block is the origin, the *last* block is the symptom you saw. The fix lives near the top: the database is refusing connections on that host and port. The `DatabaseError` is just a friendlier label your own code wrapped around it.

There is a second glue message you will see, and it is a trap: **"During handling of the above exception, another exception occurred."** That one means a *second, unrelated* exception fired inside an `except` block while handling the first — for example, your error handler itself threw a `KeyError` because it tried to log a field that did not exist. This is usually a bug *in your error handling*, not the original problem, and it is dangerous because the second exception's traceback is the one printed last and most prominent, so people fix the error handler and never see the real failure that triggered it. When you see "during handling," read the *first* traceback as the real event and treat the second as a separate, self-inflicted wound in your `except` clause.

![A matrix comparing Python, Java, Go, Node, and C++ on the order they print stack frames and where the root cause sits in each language](/imgs/blogs/reading-a-stack-trace-across-languages-2.png)

The matrix above is the cheat sheet for the whole post, and Python is row one: most recent call last, read bottom-up, the last line is where it crashed and the frames above it are where the bug came from. Pin that table somewhere; the next four sections fill in the other rows, and every one of them disagrees with Python about *something*.

## 3. Java and the JVM: read top-down, but the root is the last "Caused by"

The JVM does the opposite of Python, and the disagreement is the number-one reason engineers misread Java traces. Here is a typical one:

```java
Exception in thread "http-nio-8080-exec-3" javax.servlet.ServletException: Handler dispatch failed
    at org.springframework.web.servlet.DispatcherServlet.doDispatch(DispatcherServlet.java:1056)
    at com.acme.web.OrderController.placeOrder(OrderController.java:48)
    at com.acme.web.OrderController.checkout(OrderController.java:31)
    ... 14 more
Caused by: org.springframework.dao.DataAccessResourceFailureException: could not get pool connection
    at org.springframework.jdbc.support.SQLExceptionTranslator.translate(SQLExceptionTranslator.java:72)
    at com.acme.repo.OrderRepository.save(OrderRepository.java:64)
    ... 22 more
Caused by: java.sql.SQLException: Connection is not available, request timed out after 30000ms
    at com.zaxxer.hikari.pool.HikariPool.getConnection(HikariPool.java:213)
    ... 28 more
Caused by: java.net.ConnectException: Connection refused (Connection refused)
    at java.base/java.net.PlainSocketImpl.socketConnect(Native Method)
    at com.acme.db.Driver.connect(Driver.java:88)
    ... 31 more
```

The JVM prints **most recent call first**: the top line is the exception that was thrown closest to where you saw the failure, and the `at ...` frames below it read from innermost toward outermost. So far that is just Python upside down. The real lesson is the **"Caused by:" chain**. Each `Caused by:` block is an exception that was caught and wrapped by the block above it. The servlet layer caught a `DataAccessResourceFailureException` and wrapped it in a `ServletException`; the data-access layer caught a `SQLException` and wrapped it; the connection pool's `SQLException` was caused by a `ConnectException`. Reading them top to bottom, you are walking from the *most abstract, highest-level* description of the failure down toward the *most concrete, lowest-level* cause.

The rule is simple and you should tattoo it on the inside of your eyelids: **the root cause is the LAST "Caused by:" block.** Not the first exception, which is the polite high-level wrapper your framework produced. The last `Caused by:` here is `java.net.ConnectException: Connection refused`, and the frame under it that belongs to *your* code is `com.acme.db.Driver.connect(Driver.java:88)`. That is the real story: your database driver tried to open a socket to the database host, and the host refused the connection — nothing is listening on that port, or a firewall dropped it, or you pointed at the wrong host. Everything above it — the pool timeout, the SQL exception, the data-access exception, the servlet exception — is the *consequence* of that one refused socket, dressed up in four progressively friendlier costumes as it unwound through the framework layers.

![A branching graph showing a Java exception that splits into a wrong path where the engineer fixes the top servlet error and a correct path down four Caused-by layers to a refused database connection](/imgs/blogs/reading-a-stack-trace-across-languages-4.png)

The graph above shows the trap and the cure side by side. Read only the top exception and you "fix" the servlet — add a try/catch, return a 503, maybe blame Spring — and the bug comes right back, because the database is still refusing connections. Keep reading down the chain and the last `Caused by:` hands you the actual fault. I have watched senior engineers waste an hour on the top exception because the bottom of a Java trace is *visually* the least prominent — it is indented, it is full of framework frames, it ends in `... 31 more`. Train yourself to scroll straight to the bottom of a Java trace, find the last `Caused by:`, and read *that* exception first. Then scan its frames for the first one in *your* package (`com.acme.*`), which is the line you actually need to look at.

One more Java-specific detail: that **`... 22 more`** at the end of each block is not truncation you should fear. It means "the remaining 22 frames are identical to the frames already shown in the enclosing exception, so I am not repeating them." It is a space-saving optimization, and the omitted frames are the *common tail* of the stack (the servlet container, the thread pool, the run loop) that every layer shares. You almost never need them. Do not confuse this friendly `... N more` with a genuinely truncated trace, which is a different problem we will hit in section 8.

#### Worked example: the NullPointerException whose real cause is four levels down

Let me make the Caused-by skill concrete with the kind of bug that eats an afternoon. The pager fires: a flood of 500s, grouped under `java.lang.NullPointerException`. The first engineer reads the top of the trace — a `NullPointerException` thrown inside the JSON serializer while rendering the response — and concludes "the serializer can't handle a null field, let me add a `@JsonInclude(NON_NULL)`." They ship it. The 500s drop... for ten minutes, because now instead of crashing the serializer just silently omits a field, and downstream a *different* service starts failing on the missing field. The bug moved; it did not die.

The real trace had four `Caused by:` blocks. The top `NullPointerException` was in the serializer. The first `Caused by:` was an `EntityNotFoundException` from the ORM. The second was a `DataIntegrityViolationException`. The *last* `Caused by:`, at the very bottom under three layers of `... N more`, was a `SQLException: deadlock detected` from the connection pool, originating in `com.acme.repo.LedgerRepository.debit(LedgerRepository.java:71)`. The actual story: two concurrent checkout requests grabbed row locks in opposite order, the database picked one as the deadlock victim and rolled it back, the ORM saw a missing entity (because the rollback erased the row it expected), returned `null`, and *that* null reached the serializer four frames later and 200 milliseconds of unwinding away. The `NullPointerException` on top was the symptom of a symptom of a symptom. The fix — order the lock acquisition consistently so the deadlock can never form, see the database-locking discussion in [debugging distributed systems in production](/blog/software-development/microservices/debugging-distributed-systems-in-production) — was 80 lines away from the line the trace crashed on. The measurement that proved it: with consistent lock ordering deployed, the deadlock counter in the database went from roughly 40 per hour at peak to zero over a 72-hour window, and the NPE group in the error tracker dropped from about 9,000 events/day to none. You only find that fix by reading to the *last* Caused-by; the top exception would have had you tuning the serializer forever.

## 4. Go: panics, goroutine dumps, and finding the one that owns the crash

Go does not have exceptions; it has **panics**, and its trace format reflects a different runtime model. Here is a panic from a small Go program:

```go
panic: runtime error: index out of range [3] with length 3

goroutine 1 [running]:
main.processBatch(0xc0000b4000, 0x3, 0x3)
	/app/batch.go:27 +0x1a5
main.main()
	/app/main.go:14 +0x6f
exit status 2
```

Read it top to bottom. The first line is the panic value and message: an index-out-of-range, trying to read index 3 of a slice with length 3 (valid indices are 0, 1, 2 — this is a classic fence-post error, reaching one past the end). Then comes **`goroutine 1 [running]:`** — Go organizes the dump *by goroutine*, and the `[running]` status marks the one that actually panicked. The frames below it read **most recent first**, like Java: `main.processBatch` at `batch.go:27` is the innermost frame where the panic fired, and `main.main` at `main.go:14` is its caller. The hex numbers after the function name are the argument values in registers/stack (here `0x3, 0x3` are the slice's length and capacity), and the `+0x1a5` is the byte offset into the function — useful for `addr2line`-style work but usually ignorable. So in Go you read the *panicking goroutine's* frames from the top, find the first one in `main` or your package, and that is your crash site; walk down toward `main` to find the bug.

The thing that makes Go traces hard is that real servers have *thousands* of goroutines, and when a program panics — or when you send it a `SIGQUIT` to dump every goroutine — you get a wall of them:

```go
goroutine 42 [running]:
main.(*Worker).handle(0xc000180000)
	/app/worker.go:88 +0x2f1
...

goroutine 1 [chan receive, 9 minutes]:
main.main()
	/app/main.go:52 +0x118

goroutine 1843 [select]:
net/http.(*persistConn).readLoop(0xc0003a4000)
	/usr/local/go/src/net/http/transport.go:2210 +0xda5

goroutine 1844 [IO wait]:
internal/poll.runtime_pollWait(0x7f..., 0x72)
	/usr/local/go/src/runtime/netpoll.go:343 +0x85
```

The skill here is the **goroutine status in brackets**. `[running]` is the one on a CPU right now — if there is a panic, this is the culprit, and you read *its* frames. `[chan receive, 9 minutes]` means this goroutine has been blocked sending or receiving on a channel for nine minutes — a huge red flag for a deadlock or a stuck pipeline, because nine minutes is forever. `[IO wait]` and `[select]` are usually healthy waiting. `[semacquire]` means it is blocked on a mutex. When you are debugging a *hang* rather than a panic, you do not have a `[running]` goroutine to anchor on — instead you scan for the long durations and the `[chan receive]` / `[semacquire]` / `[sync.Mutex.Lock]` states, because a hung Go program is a graph of goroutines all waiting on each other. To get this full dump from a *running, hung* process, send it `SIGQUIT`:

```bash
# Find the process and send SIGQUIT to dump every goroutine's stack to stderr.
$ kill -QUIT $(pgrep -f myserver)

# Or, if you set GOTRACEBACK=all, you get all goroutines on any crash:
$ GOTRACEBACK=all ./myserver
# GOTRACEBACK=system also shows runtime goroutines; default 'single'
# only shows the panicking one, which hides the goroutine that owns the bug.
```

That `GOTRACEBACK=all` is the single most useful Go trace flag and almost nobody sets it. The default, `single`, prints only the panicking goroutine — which is exactly the thread B problem from section 1: the goroutine that *computed* the bad value, or the one holding the lock that everyone else is blocked on, is invisible. Set `GOTRACEBACK=all` in your deployment environment and a panic dumps every goroutine, so you can find the one that owns the resource everyone is waiting for. For a deeper treatment of channel-blocked goroutine graphs, the sibling post on [debugging async and event loops](/blog/software-development/debugging/debugging-async-and-event-loops) walks through how concurrency runtimes lose the logical caller.

One Go-specific subtlety bites people coming from exception languages: a panic in one goroutine that is *not recovered* crashes the *entire process*, not just that goroutine. There is no per-goroutine error boundary unless you write a `recover()` in a deferred function on that goroutine. So a stray `nil` map write in a background worker takes down the whole server, and the trace you get is that worker's — frequently a goroutine spawned deep inside a library, with a stack that bottoms out in `created by some/library.(*Pool).worker` rather than in your `main`. That `created by` line at the very bottom of a goroutine's stack is gold: it tells you *where the goroutine was launched*, which is the closest thing Go gives you to "who is responsible for this goroutine," and it is often the only link back to your code in an otherwise all-library stack.

#### Worked example: the Go panic whose owner was a different goroutine

A real shape this takes: a Go service started crashing a few times a day with `panic: assignment to entry in nil map`, and the default trace pointed at `cache.go:53` inside `(*Store).Set` on the `[running]` goroutine. The obvious read — "the map is nil, add a nil check in `Set`" — was wrong, because `Set` could only be reached after `New()` allocated the map, and `New()` clearly ran. Setting `GOTRACEBACK=all` and catching the next crash told the real story: there were *two* goroutines touching the store. One was running `(*Store).Set` (the panicking one). Another, blocked in `[chan receive, 4 minutes]`, was a `reload()` goroutine that, on a config change, replaced the store by assigning a *freshly-constructed, not-yet-initialized* `Store` value into the shared pointer — and for a few microseconds its inner map was `nil` because the constructor set the map *after* publishing the pointer. The `Set` goroutine raced into that window. The crash goroutine was innocent; the bug was the publish-before-initialize ordering in the `reload` goroutine, visible only because the full dump showed *both* goroutines and the `created by config.(*Watcher).watch` line told us where `reload` was launched. The fix was to fully construct the new store and only then atomically swap the pointer (publish-after-initialize). The measurement: with `sync/atomic.Value` doing the swap, Go's race detector (`go test -race`) went from flagging the unsynchronized publish on roughly one in fifty runs to clean across 5,000 runs, and production panics went from about three per day to zero over two weeks. The whole diagnosis depended on the one flag that shows every goroutine, not just the one Go chose to print.

## 5. Node and JavaScript: the async trace that looks empty

JavaScript's stack trace problem is the worst of any language here, and it is entirely caused by the event loop. Synchronous JavaScript reads like Java — most recent call first, top-down:

```js
TypeError: Cannot read properties of undefined (reading 'total')
    at calculateTax (billing.js:42:18)
    at buildInvoice (billing.js:30:22)
    at main (app.js:8:14)
    at Object.<anonymous> (app.js:15:1)
```

That is fine: `calculateTax` at `billing.js:42` tried to read `.total` of something `undefined`, and you walk up to find who passed the undefined value. The trouble starts with **`async`/`await`**. Consider this:

```js
async function getOrder(id) {
  const user = await fetchUser(id);   // suspends here
  return user.account.total;          // throws if account is undefined
}

async function handleRequest(req) {
  return await getOrder(req.id);
}
```

When `getOrder` hits `await fetchUser(id)`, the function *suspends* and returns control to the event loop. The synchronous call stack — the one with `handleRequest` and the HTTP server frames on it — **unwinds completely**, because the event loop has to go run other work while the network request is in flight. Later, when `fetchUser`'s promise resolves, the event loop schedules the *continuation* of `getOrder` to run, and it runs on a **fresh, nearly empty stack**. If `user.account` is undefined and the next line throws, the default V8 trace you get is:

```js
TypeError: Cannot read properties of undefined (reading 'total')
    at getOrder (order.js:3:22)
    at processTicksAndRejections (node:internal/process/task_queues:95:5)
```

Two frames. `getOrder` and an internal Node task-queue frame. **`handleRequest` is gone. `main` is gone. The HTTP handler is gone.** The trace cannot tell you which request triggered this, because at the moment of the throw, none of those frames were on the stack — they unwound at the `await` minutes (or milliseconds) ago. This is the mechanism behind every "the trace is useless" complaint about Node: the call stack literally does not contain the logical caller, because the logical caller already returned to the event loop.

![A left-to-right timeline showing a handler calling fetchUser, hitting await and unwinding the stack, the event loop running other work, the promise resolving on a new stack, the throw with the caller gone, and async traces restoring it](/imgs/blogs/reading-a-stack-trace-across-languages-5.png)

The timeline above is the sequence: call, await, unwind, loop runs other work, resolve on a new stack, throw with no caller, then — the fix — async traces stitch the caller back in. The cure is **async stack traces**, which V8 supports by recording, at each `await`, a link back to the suspended frame so the runtime can reconstruct the logical chain. In modern Node (12+) it is on by default for `async`/`await`, but it is easy to lose it, and you can force the richer behavior:

```bash
# Force async stack traces on (default in Node 12+, but be explicit on older runtimes):
$ node --async-stack-traces app.js

# For deeper non-await async (callbacks, timers, custom promises), bump the limit
# and use async_hooks-based tooling; raise the captured frame count too:
$ node --stack-trace-limit=50 app.js
```

With async traces on, the same crash prints the logical chain restored across the await boundaries:

```js
TypeError: Cannot read properties of undefined (reading 'total')
    at getOrder (order.js:3:22)
    at async handleRequest (order.js:8:10)
    at async Server.<anonymous> (server.js:21:5)
```

Now you can see it: `handleRequest` called `getOrder`, the server called `handleRequest`, and you know exactly which request path produced the undefined account. The `async` marker on each frame tells you it was reconstructed across an await, not present on the raw synchronous stack.

![A two-column before-and-after contrast showing a default Node trace with one anonymous frame versus an enabled async stack trace showing the full caller chain down to the source line](/imgs/blogs/reading-a-stack-trace-across-languages-7.png)

The before-after above is the payoff: an empty-looking trace with one `<anonymous>` frame becomes a full caller chain once you enable async stack traces. There are two more Node-specific reading hazards. First, **`<anonymous>`** frames: an arrow function or callback with no name shows up as `<anonymous>`, which tells you a location (`file:line:col`) but not a function name — you have to open that file and line to see which closure it is. Name your functions and this problem largely disappears. Second, **minified code**: in production the trace points at `app.min.js:1:48211`, a single line because the whole bundle was minified to one line. That column number plus a **source map** lets you recover the original location:

```bash
# A source map (app.min.js.map) maps minified positions back to original source.
# Tools like 'source-map' or your error tracker (Sentry, etc.) do this automatically;
# manually you can resolve a single frame:
$ npx source-map-cli resolve app.min.js.map 1 48211
#   -> original: src/billing/calculateTax.js:42:18
```

Without the source map, a minified production trace is as useless as a stripped C binary — which is the next section, because the symbolization problem is the same problem in a different costume.

#### Worked example: the Node async trace that was empty until I turned on the flag

A real one. A payments service in Node started logging `UnhandledPromiseRejection: TypeError: Cannot read properties of null (reading 'id')` a few hundred times an hour, always at `at async (transfer.js:51:30)` and nothing else — one frame, no caller, no request context. The team had no idea which of a dozen entry points triggered it, and it reproduced maybe one in ten thousand requests, so adding logs everywhere was expensive and slow. The first move was not more logging; it was getting a *better trace*. They were on Node 16 but had `--stack-trace-limit=10` set low and async traces effectively truncated under load. They redeployed with `node --async-stack-traces --stack-trace-limit=50` and within twenty minutes the trace filled out: `transfer.js:51` was called from `async settleBatch (batch.js:88)`, called from `async cronTick (scheduler.js:33)`. It was not a user request at all — it was the *nightly batch settlement job*, which fetched an account that had been soft-deleted (returning `null`) and tried to read its `.id`. Once the caller chain was visible, the bug was obvious and ten minutes from fix: the batch query needed to filter out soft-deleted accounts. The measurement: the rejection count went from roughly 300/hour to zero over the next two nightly runs, and the fix was confirmed causal by re-enabling the unfiltered query path in staging and watching the exact rejection reappear at the exact rate. The entire investigation hinged on one runtime flag that turned a one-frame trace into a real caller chain. The trace was never useless; it was *unconfigured*.

## 6. C and C++: symbols, `??`, inlined frames, and optimized-out variables

Native code is where stack traces get genuinely hard, because the trace you get depends on how the binary was *compiled*, and a release build will happily hand you garbage. Start with the easy case — a debug build crashing under `gdb`:

```bash
$ gdb ./app core
(gdb) bt
#0  0x0000555555555189 in scan_token (s=0x0) at parse.c:88
#1  0x00005555555552d4 in parse_line (line=0x5555557592a0 "GET /") at parse.c:142
#2  0x0000555555555401 in handle_request (fd=7) at server.c:60
#3  0x00005555555554f8 in main (argc=1, argv=0x7fffffffe3c8) at server.c:91
```

This is a *good* trace. `gdb`'s `bt` (backtrace) prints **innermost first**: `#0` is the crash frame, `scan_token` at `parse.c:88`, and crucially it shows the argument `s=0x0` — a null pointer was passed in. Walk up: `#1 parse_line` called it with `line` pointing at the string `"GET /"`, `#2 handle_request` with `fd=7`, `#3 main`. The argument values *in the frames* are the gift here: `s=0x0` at frame 0 tells you `scan_token` received null, and `line="GET /"` at frame 1 tells you the line being parsed was fine, so the bug is whatever computed the `s` argument inside `parse_line` between getting `line` and calling `scan_token`. The frame arguments turned "it segfaulted somewhere in parsing" into "a null was passed to `scan_token` from `parse_line`," which is a one-function search.

Now the *bad* case — the same crash in a stripped release binary:

```bash
$ gdb ./app core
(gdb) bt
#0  0x00007f3c8a4b2189 in ?? ()
#1  0x00007f3c8a4b22d4 in ?? ()
#2  0x0000564f2a1c1401 in ?? ()
#3  0x0000564f2a1c14f8 in ?? ()
```

Every frame is **`?? ()`** — the binary was stripped of its symbol table, so `gdb` has addresses but no names. This is the C/C++ equivalent of the minified Node trace: the information to translate addresses into functions exists, it is just not in the binary. **Symbolization** is the process of mapping those raw addresses back to file and line, and you do it with the debug symbols that the build either kept separately or that you can regenerate. If you saved a `.debug` file or built with `-g` into a separate symbol file, you point `addr2line` at it:

```bash
# Translate a raw address from a stripped binary back to file:line using
# the separate debug-symbols file (built with -g, split out via objcopy):
$ addr2line -e app.debug -f -C 0x1189
scan_token
/src/parse.c:88

# Symbolize a whole backtrace at once by piping the addresses:
$ addr2line -e app.debug -f -C 0x1189 0x12d4 0x1401 0x14f8
```

![A two-column before-and-after contrast showing a stripped binary printing address-only question-mark frames versus a symbolized trace resolving an address to a file and line with addr2line](/imgs/blogs/reading-a-stack-trace-across-languages-3.png)

The before-after above is the symbolization story: `?? ()` frames become `parse.c:88 in scan_token` once you feed the addresses and the debug file through `addr2line`. The practical lesson — and this is a build-pipeline decision, not a debugging trick — is **keep your symbols.** The standard professional move is to build with `-g` (full debug info), then split the symbols into a separate file and ship the stripped binary to production, keeping the symbol file in your artifact store keyed by the binary's **build-id**:

```bash
# Build with debug info, then separate symbols from the shipped binary:
$ gcc -g -O2 -o app app.c
$ objcopy --only-keep-debug app app.debug      # symbols go here
$ objcopy --strip-debug app                     # shipped binary is small
$ objcopy --add-gnu-debuglink=app.debug app     # link them by build-id
```

Now production runs the small stripped binary, but when it cores, you grab the matching `app.debug` from your artifact store (matched by build-id so you never symbolize against the wrong version) and get a perfect trace. On macOS the same role is played by **`.dSYM`** bundles; on Windows, **`.pdb`** files. Same idea everywhere: the symbols are precious, store them keyed to the build, symbolize after the fact.

But there is a deeper C/C++ hazard that even good symbols do not fully fix: **inlining and optimization**. At `-O2`, the compiler may **inline** a small function directly into its caller, so the function literally does not exist as a separate frame at runtime — its body was pasted into the caller. The trace then shows the caller's frame where you expected the callee, and a naive reading sends you to the wrong function. Modern `gdb` and `addr2line` partly compensate by reading the DWARF debug info that records inlining, and will show inlined frames with a marker:

```bash
(gdb) bt
#0  0x... in validate_utf8 (inlined into scan_token) at utf8.c:30
#1  0x... in scan_token (s=0x0) at parse.c:88
```

That `(inlined into ...)` is your tell. Without it (older tooling, missing debug info) you would see `scan_token` at the line where `validate_utf8` was inlined and be confused about why `parse.c:88` "calls" a UTF-8 validator. Worse, at `-O2` a variable you try to print may report **`<optimized out>`**:

```bash
(gdb) info args
s = 0x0
(gdb) print len
$1 = <optimized out>
```

`<optimized out>` means the compiler kept that value only in a register that got reused, or computed it lazily, so its value is genuinely not recoverable at this point — the information was optimized away to make the code faster. This is the deep reason for a rule that runs through this whole series: **if you are chasing a hard crash, reproduce it at `-O0` first.** A `-O0` build keeps every function as its own frame, keeps every variable live, and inlines nothing, so the trace and the variables tell the truth. The cost is that some bugs — especially undefined-behavior bugs and timing-sensitive races — *change or vanish* at `-O0` (a "heisenbug" — a bug that disappears when you observe it), in which case you are stuck symbolizing the optimized build carefully and reading the `(inlined into ...)` markers. But for the common case, ten minutes spent rebuilding at `-O0` saves an hour of squinting at `<optimized out>`.

## 7. Walking the frames live: turning a trace into a hypothesis

A printed stack trace is a *snapshot*; a live debugger lets you *walk* the frames and interrogate each one, which is how you go from "where it crashed" to "what value was wrong and where it came from." The two essential moves in any debugger are **up** (move to the caller's frame) and **down** (move back toward the crash), plus printing the arguments and locals in whichever frame you are standing in. Here is a full `gdb` session walking a null-pointer crash:

```bash
$ gdb ./app core
(gdb) bt full              # backtrace WITH locals in every frame
#0  scan_token (s=0x0) at parse.c:88
        c = 0 '\000'
        len = 0
#1  parse_line (line=0x...) at parse.c:142
        tok = 0x0
        field = 0x555555759310 "GET"
#2  handle_request (fd=7) at server.c:60
        buf = "GET /index.html HTTP/1.1\r\n..."
        n = 26

(gdb) frame 1              # move to frame #1, the caller of scan_token
(gdb) info args           # what arguments did parse_line receive?
line = 0x555555759310 "GET /index.html HTTP/1.1"
(gdb) info locals         # and its local variables?
tok = 0x0
field = 0x555555759310 "GET"
(gdb) print tok           # tok is null — and tok was passed to scan_token as s
$1 = (char *) 0x0
(gdb) list 138,143        # show the source around the call site
138     char *tok = strtok(line, " ");
139     // ... advance past method
140     tok = next_token(tok);   // returns NULL at end of line
141
142     return scan_token(tok);  // passes NULL straight through
```

This is the whole craft in one screen. `bt full` gave us the backtrace *with every frame's locals*, which immediately shows `tok = 0x0` in frame 1 — the variable that became the null `s` in frame 0. We moved `up` to `frame 1`, read `info args` and `info locals`, confirmed `tok` was null, and `list`ed the source to see that `next_token` returned NULL at end of line and the code passed it straight into `scan_token` with no check. The bug is `parse.c:142`: a missing null check after `next_token`. We never guessed; we read the values out of the frames and let them name the line.

![A grid showing a gdb session walking from the bt full listing through frame zero, moving up and down to inspect args and locals, and arriving at the missing-initialization root cause](/imgs/blogs/reading-a-stack-trace-across-languages-8.png)

The grid above is that walk laid out as a path: `bt full` lists the frames, frame 0 shows the crash with a null pointer, you move up and read `info args` to find a null that should not be null, move down to recheck locals, and arrive at the root cause — a value that was never set. The same moves exist in Python's `pdb`, which is worth showing because the verbs differ slightly:

```python
# Drop into a post-mortem debugger right where the exception fired:
$ python -m pdb app.py
# ... or in code: import pdb; pdb.post_mortem()  after catching the exception

(Pdb) where          # the 'bt' equivalent: print the stack
  app.py(34)main()
-> cfg = load_config("settings.yaml")
  config.py(18)load_config()
-> return parse_timeout(raw["timeout"])
> config.py(9)parse_timeout()
-> return int(value) * 1000
(Pdb) up             # move to the caller, load_config
> config.py(18)load_config()
-> return parse_timeout(raw["timeout"])
(Pdb) args           # what arguments did load_config get?
path = 'settings.yaml'
(Pdb) p raw['timeout']   # print the offending value
'30s'
(Pdb) down           # back toward the crash
> config.py(9)parse_timeout()
-> return int(value) * 1000
(Pdb) p value
'30s'
```

`where` is `bt`, `up`/`down` move between frames, `args` prints the current frame's arguments, and `p expr` evaluates any expression in the current frame's scope. The discipline is identical to `gdb`: stand in a frame, read what it received, decide whether *this* frame got bad input (so the bug is above) or produced bad output (so the bug is here). The frame sequence plus the argument values *is* the hypothesis: "`scan_token` got null because `parse_line` passed `tok`, which `next_token` returned as null at end-of-line, and nobody checked." That sentence is what a trace is *for*. The figure-1 idea — the bug lives up the stack from the crash — is exactly what `up` is for.

There is one Python tool worth its own mention because it rescues a class of "no trace at all" crashes: **`faulthandler`**. A pure-Python exception always prints a traceback, but a *hard crash* — a segfault inside a C extension, an infinite C loop, or a hang — kills the interpreter with no Python traceback at all. `faulthandler` installs handlers for fatal signals (and can dump on a timeout) so even a C-level crash prints the *Python* stack that led into it:

```python
import faulthandler
faulthandler.enable()              # dump Python traceback on SEGV/FPE/etc.
faulthandler.dump_traceback_later(60, repeat=True)  # also dump if hung 60s

# Or from the command line, no code change:
#   $ python -X faulthandler app.py
#   $ PYTHONFAULTHANDLER=1 python app.py
```

That `dump_traceback_later` is the move for a *hang*: it prints the Python stack of every thread every 60 seconds, so a process that is stuck — not crashed, just frozen — finally tells you which line every thread is wedged on. It is the Python analogue of sending `SIGQUIT` to a Go process for a goroutine dump, and it is the difference between "the job hangs and I have no idea where" and "all three workers are blocked on the same lock at `queue.py:88`."

## 8. The frames that lie: async, inlined, truncated, and wrong-thread traces

By now you can read a trace in five languages and walk it live. The last skill is the hardest and the most senior: knowing when the trace is *lying to you*, because a trace that is *technically accurate* can still point you at the wrong thing. Here are the five lies, each with its tell and its fix.

**Lie 1 — the async stack that dropped the logical caller.** We covered the mechanism in section 5: at an `await` (Node), an `await`/`asyncio` suspension (Python), or a future's `.then` (everywhere), the synchronous stack unwinds, so the resumed code throws on a stack that does not contain who scheduled it. The *tell* is a suspiciously short trace that ends in a runtime scheduler frame (`processTicksAndRejections`, `asyncio/events.py`, `tokio` internals) with no application caller. The *fix* is to enable the runtime's async-stack support (Node `--async-stack-traces`, Python 3.11+ has improved async traces, `asyncio` debug mode via `PYTHONASYNCIODEBUG=1`) so the runtime stitches the awaited frames back. This is important enough that the sibling post [debugging async and event loops](/blog/software-development/debugging/debugging-async-and-event-loops) is dedicated to it; the one-line summary is that the call stack is the wrong data structure for "who logically caused this" in async code, and you need causal stack tracking to recover it.

**Lie 2 — the inlined or tail-call-eliminated frame.** At `-O2` the compiler inlines small functions (their frame vanishes, covered in section 6) and performs **tail-call elimination**: when a function's last act is to call another function, the compiler can reuse the current frame instead of pushing a new one, so the *caller's* frame is gone from the trace. The tell is a trace where you *know* function A called B but A is not in the stack — A tail-called B and got optimized away. The fix is the same as inlining: read the `(inlined into ...)` markers if your tooling shows them, disable tail-call optimization while debugging if your compiler allows it, or reproduce at `-O0`. This is why a recursive function that should show 5,000 frames sometimes shows 3: every recursive tail call reused the same frame.

**Lie 3 — the truncated trace, "... 23 more" / depth limits.** Real traces get cut off. Java's `... 23 more` is benign (shared common tail, section 3), but a *genuine* truncation happens when a runtime caps stack-trace depth: Node defaults to `Error.stackTraceLimit = 10`, Python has `sys.tracebacklimit`, the JVM can elide repeated frames. A deeply recursive crash or a long async chain gets cut, and the frame you need is past the cut. The tell is a trace that ends abruptly at a round number (exactly 10 frames in Node) or in the middle of a recursion. The fix is to raise the limit *before* reproducing: `Error.stackTraceLimit = 50` in Node, `node --stack-trace-limit=100`, leave `sys.tracebacklimit` at its default (do not set it low), and for native code increase `gdb`'s `backtrace limit`. A truncated trace is not a dead end; it is an under-configured one.

**Lie 4 — framework frames drowning your frames.** A real production trace is 90% noise: the web framework's dispatch chain, the ORM's proxy frames, the runtime's scheduler, dependency-injection plumbing. Thirty frames, of which three are yours. The tell is just length — and the danger is grabbing the first *recognizable* word, which is usually a framework class, and debugging *it*. The fix is a filtering discipline: scan for the first frame whose package or path is *your* code (`com.acme.*`, `src/`, not `node_modules/` or `site-packages/`), and start there.

![A tree separating a thirty-frame trace into runtime frames, framework frames, and the three frames of your own code where the bug actually lives](/imgs/blogs/reading-a-stack-trace-across-languages-6.png)

The tree above is that filtering move made visual: a 30-frame trace splits into runtime frames (scheduler, GC — ignore), framework frames (dispatch, filters — rarely the bug), and *your* frames (3 of 30 — the bug is in `handler.save()`). Most error trackers can do this for you: configure "in-app frames" (Sentry calls it `in_app`) so your code is highlighted and `node_modules`/`site-packages` are collapsed by default. The skill is the same with or without tooling: the bug is almost always in the deepest frame that you wrote.

**Lie 5 — the trace from the wrong thread.** This is the subtlest and the one I have seen waste the most senior-engineer hours. As section 1 explained, the trace you get is the stack of the thread that *threw*, and in a multithreaded program the thread that *created* the bad value or *holds the lock everyone wants* is on a different stack entirely. A worker thread crashes dereferencing an object that a *producer* thread half-initialized and published through a queue; the crash trace shows the worker, which is innocent, and never mentions the producer, which is guilty. The tell is a trace where every frame looks correct and the crash value is impossible to produce *within those frames* — the bad value must have come from outside. The fix is to dump *all* threads, not just the crashing one: `GOTRACEBACK=all` in Go (section 4), `faulthandler.dump_traceback()` for all Python threads, `jstack <pid>` for a full JVM thread dump, `thread apply all bt` in `gdb`:

```bash
# Dump every thread's stack, not just the one that crashed:
(gdb) thread apply all bt          # gdb: backtrace for all threads
$ jstack 4821                      # JVM: full thread dump by pid
$ kill -QUIT $(pgrep myserver)     # Go: SIGQUIT dumps all goroutines
```

In a JVM thread dump you are looking for the deadlock report at the bottom (the JVM detects and prints cycles of `BLOCKED` threads waiting on each other's monitors) and for many threads `BLOCKED` on the same lock — that lock's *owner* is your bug, and it is on a thread the crash trace never showed you. For the locking and ordering mechanics behind these multithread cases, the database-locking and ordering discussion in [debugging distributed systems in production](/blog/software-development/microservices/debugging-distributed-systems-in-production) goes deeper; the trace-reading point is simply that **one thread's stack is one slice of the truth, and a concurrency bug needs all the slices.**

## 9. Getting a good trace before you need it

Everything above assumes you *have* a readable trace. Half the battle is making sure you will, because a trace you cannot symbolize is a trace you cannot read, and you find out at 3am when it is too late to rebuild. This is a *prevention* section — the last stage of the series loop — and it is mostly build-and-deploy hygiene. Here is the table I give every team:

| Language / runtime | What kills the trace | What to do *before* you need it |
| --- | --- | --- |
| C / C++ | stripped binary, `?? ()` frames | build `-g`, split symbols with `objcopy`, store `.debug` keyed by build-id |
| C / C++ | `<optimized out>`, inlined frames | keep a `-O0` repro path; read `(inlined into ...)` markers in `gdb` |
| Python | hard crash in C ext, no traceback | `faulthandler.enable()` + `dump_traceback_later` for hangs |
| Java / JVM | lost `Caused by`, swallowed exception | never `catch` without chaining (`throw new X(msg, cause)`); never empty `catch {}` |
| Node / JS | empty async trace, `<anonymous>` | `--async-stack-traces`, raise `stackTraceLimit`, name your functions |
| Node / JS | minified production trace | ship source maps; let the error tracker resolve them |
| Go | only the panicking goroutine | set `GOTRACEBACK=all`; `SIGQUIT` for a live full dump |
| Any multithreaded | wrong-thread trace | dump *all* threads (`jstack`, `thread apply all bt`) |

The single most damaging anti-pattern across all of these is the **swallowed exception**: a `catch (Exception e) {}` with an empty body, or `except: pass`, that catches the original failure and throws away the trace, then later fails somewhere else with no connection to the cause. You get a trace, but it is a trace of the *consequence*, with the root cause permanently erased. The prevention rule is absolute: **never catch an exception without either handling it meaningfully or re-raising it with the cause attached.** In Java that is `throw new ServiceException("checkout failed", e)` — note the `e`, which builds the `Caused by:` chain you learned to read in section 3. In Python it is `raise ServiceError("checkout failed") from exc`. Drop the cause and you have personally deleted the evidence that section 3 and section 5 taught you to follow.

The second prevention move is **structured context on the trace**. A bare trace tells you *what* and *where* but not *which request, which user, which input*. Attaching a correlation/request ID to every log line and to the exception means that when you get the trace, you can pull the full request context around it. This crosses into observability, which is its own discipline — [observability metrics logs traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) covers building it in deliberately — but the trace-reading point is that a stack trace plus a correlation ID is worth ten stack traces without one, because the ID lets you reproduce the exact input that produced the crash. And reproduction, per [reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging), is the step that turns a trace into a fix.

#### Worked example: the trace that pointed at the wrong line for two hours

One more, because it ties the whole post together. A C++ service segfaulted in production maybe once a day, no pattern. The core dump, symbolized against the *release* build, put the crash at `cache.cpp:204` inside `Cache::lookup`, on a line that read `return entry->value;`. Two engineers spent two hours convinced `entry` was null and adding null checks to `lookup`, which did nothing — the crashes continued. The trace was technically accurate and completely misleading, for two reasons that this post named. First, the build was `-O2`, and `cache.cpp:204` was where the compiler had **inlined** a different function, `Entry::deref`, whose body actually did the dereference — the crash was "at line 204" only because that is where the inlined code landed; the real logic lived in `Entry::deref`. The `(inlined into ...)` marker was missing because the release build's debug info was incomplete. Second, when they finally rebuilt at `-O0` with full `-g` and reproduced under `gdb` with `thread apply all bt`, they saw the real story: `entry` was *not* null — it was a **dangling pointer**, an `Entry` that a *different thread* had evicted and freed from the cache while this thread held a raw pointer to it. The crash thread was reading freed memory (a use-after-free); the guilty thread was the eviction thread, on a stack the original single-thread trace never showed. AddressSanitizer confirmed it in one run: `-fsanitize=address` printed both the read of freed memory *and* the stack of the thread that freed it, which is exactly the cross-thread information a plain trace omits. The measurement: with a proper reference-counted handle replacing the raw pointer, the crash went from roughly one per day to zero over a three-week window, and ASan ran clean on the reproducer that previously failed about one in two thousand iterations. Two hours were lost to a trace that lied in two of the five ways this post catalogs — inlining and wrong-thread — and both lies were defeated by the same two moves: reproduce at `-O0`, and dump *all* the threads.

## War story: traces that changed the world

A few real bug classes are worth naming because they show what is at stake in reading — or failing to read — a trace and the context around it.

**Heartbleed (2014).** The OpenSSL Heartbleed vulnerability was a buffer over-read: a heartbeat request claimed a payload length larger than it actually sent, and the server dutifully copied that many bytes out of its memory into the reply, leaking up to 64KB of whatever happened to be adjacent — private keys, session data, passwords. It is a stack/heap-trace lesson by contrast: the bug produced *no crash and no trace at all*, because reading slightly-too-far in memory is undefined behavior that usually "succeeds" silently. The tool that *would* have caught it pre-release is exactly the one from our worked example — AddressSanitizer flags an out-of-bounds read the instant it happens, turning a silent, traceless leak into a loud, fully symbolized stack trace pointing at the exact line. The lesson: some of the worst bugs are precisely the ones that *do not* produce a trace, and the cure is to run instrumentation (ASan, UBSan, Valgrind) that *manufactures* a trace at the moment of the violation.

**The Knight Capital deploy (2012).** Knight Capital lost about 440 million dollars in 45 minutes when a deployment left an old, repurposed feature flag active on one of eight servers, causing it to execute a dead code path that fired millions of unintended orders. There was no single stack trace that said "you will lose 440 million dollars," but the post-mortem reading of *which server*, *which flag*, *which code path* is exactly the frame-by-frame, "where did this value come from" discipline of this post applied to a deploy instead of a crash. The lesson that maps onto trace-reading: the *symptom* (runaway orders) was many layers removed from the *cause* (a flag set on one host), and only walking the chain from symptom back to origin — the Caused-by discipline, generalized to infrastructure — finds the real fault. Anyone who "read the top line" (the orders) and stopped never reaches the flag.

**The leap-second cascades (2012, 2015).** When a leap second was inserted, a number of Linux systems hit a kernel bug where the extra second triggered a livelock in the high-resolution timer code, spiking CPU to 100% across fleets simultaneously. Engineers who pulled a stack trace (or a `perf` profile) saw threads spinning in timer/futex code — frames that looked like a busy lock, not an obvious "leap second" sign. The trace told you *where* the CPU was burning (the timer subsystem) but not *why* (a clock discontinuity the code did not expect). The lesson: a trace localizes the *where*; the *why* often requires correlating the trace with an external event (a clock change, a deploy, a traffic spike) that no single frame mentions. For the anatomy of how these cascade across a fleet, [anatomy of an outage lessons from real postmortems](/blog/software-development/system-design/anatomy-of-an-outage-lessons-from-real-postmortems) is the systems-level companion to this frame-level post.

The thread through all three: a stack trace is necessary but not sufficient. It tells you *where the program was* when it broke, with high precision. It does not tell you *why* unless you read it correctly — bottom-up or top-down per language, all the way down the Caused-by chain, across threads, past the inlined and async lies — and then correlate the *where* with the *what changed* in the world around it.

## How to reach for this (and when not to)

Reading a trace well is fast and cheap, so the bias should be toward reading it *completely* before doing anything else. But there are judgment calls about how far to go.

**Always read the whole trace first.** Before you restart anything, before you add a log line, before you form a theory: read the trace top to bottom (or bottom to top, per language), find the last `Caused by:`, identify your deepest frame, and read the argument values if you have them. This takes 60 seconds and prevents the most common 3am mistake — fixing the top-line symptom. The engineer who restarted the pods in the intro skipped this step.

**Reach for a live debugger when the trace is not enough.** A printed trace gives you frames and (sometimes) arguments. When you need *locals*, *heap state*, or to walk `up`/`down` interrogating values, attach `gdb`/`pdb`/`delve`. But — and this is the senior judgment — **do not attach an interactive debugger to a latency-sensitive production process.** Breaking on `gdb` freezes the process; do that to the payments service under load and you have turned a crash into an outage. Use a core dump (post-mortem, the process is already dead, attach to the corpse) or a non-stopping snapshot instead. The sibling post on reading a core dump post-mortem covers the "the process already died, now what" path — link it by its planned slug `reading-a-core-dump-post-mortem-analysis` when it ships.

**Do not chase a heisenbug at `-O2`.** If a crash only appears in the optimized release build and the trace is full of `<optimized out>` and missing inlined frames, your first move is to *try* to reproduce at `-O0`. If it reproduces, debug there with truthful frames and variables. If it *vanishes* at `-O0`, you have learned something important — it is an undefined-behavior or timing bug — and the right tool is a sanitizer (ASan/UBSan/TSan), not more squinting at the optimized trace.

**Do not over-invest in symbolizing a one-off.** If a binary crashed once, will never run again, and you have no symbols, sometimes the honest move is "we cannot symbolize this; add `-g` and symbol storage to the build and wait for it to recur." Spending an afternoon reverse-engineering `?? ()` frames for a bug that will reproduce tomorrow with a proper build is a poor trade. Fix the build pipeline (the prevention table in section 9), then catch it cleanly next time.

**When a well-placed log answers it, do not open a debugger.** If reading the trace gives you a clear hypothesis ("`tok` is null when the line has no second token"), the fastest confirmation is often one log line or one assertion at the suspect spot, not a full debugger session. Reach for the heavy tools when the cheap read leaves you genuinely uncertain about which frame holds the bug.

## Key takeaways

- **The crash site is not the bug site.** The innermost frame is where the program *detected* a bad value; the bug is usually two or more frames up where your code *produced or passed* it. Walk up from the crash.
- **Know your language's reading direction.** Python prints most-recent-call last (read bottom-up); Java/JVM and Go and Node print most-recent first (read top-down). Starting at the wrong end is the most common misread.
- **Follow the Caused-by chain to the end.** In Java the root cause is the *last* `Caused by:`; in Python the *first* traceback under "direct cause of the following exception" is the origin. The top exception is the friendly wrapper, not the fault.
- **Async stacks drop the logical caller** because the synchronous stack unwinds at the `await`. Enable async stack traces (`--async-stack-traces`, `GOTRACEBACK=all`, `faulthandler` for hangs) to recover who actually called.
- **Keep your symbols.** Build with `-g`, split symbols with `objcopy` keyed by build-id (or `.dSYM`/`.pdb`/source maps), and store them. A stripped trace of `?? ()` frames is unreadable, and you find out at the worst time.
- **Watch for the lies:** inlined and tail-call-eliminated frames at `-O2`, truncated traces at the default `stackTraceLimit`, framework frames drowning your three, and the trace that came from a different thread than the one that owns the bug.
- **Read the argument values in the frames.** `gdb bt full`, `info args`/`info locals`, `pdb where`/`up`/`down`/`args` turn "it crashed somewhere" into "this exact value was wrong and came from this exact line." The frame values *are* the hypothesis.
- **Dump all the threads, not just the crashing one.** A concurrency bug puts the guilty party on a stack the crash trace never shows; `thread apply all bt`, `jstack`, and `GOTRACEBACK=all` reveal it.
- **Never swallow an exception.** Empty `catch {}` / `except: pass` deletes the evidence. Always re-raise with the cause attached so the Caused-by chain survives.
- **Reproduce at `-O0` before fighting an optimized trace.** Truthful frames and live variables beat an afternoon of `<optimized out>`; if the bug vanishes at `-O0`, reach for a sanitizer.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the series intro and the observe→reproduce→hypothesize→bisect→fix→prevent loop this post's *observe* stage feeds.
- [Reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — turning the trace plus a correlation ID into a deterministic repro you can attack.
- [Debugging async and event loops](/blog/software-development/debugging/debugging-async-and-event-loops) — the deep dive on why async stacks lose the caller and how causal stack tracking recovers it (sibling post).
- *Reading a core dump: post-mortem analysis* — the companion for "the process already died, attach to the corpse" (planned sibling, slug `reading-a-core-dump-post-mortem-analysis`).
- [Observability: metrics, logs, traces by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) — attaching correlation IDs and context so a trace tells you *which* request crashed.
- [Debugging distributed systems in production](/blog/software-development/microservices/debugging-distributed-systems-in-production) — cross-service Caused-by chains, deadlock ordering, and wrong-thread traces at fleet scale.
- The GDB manual (`backtrace`, `frame`, `info args/locals`, `thread apply all bt`) and the AddressSanitizer wiki — the canonical references for walking frames and for manufacturing a trace at the moment of a memory violation.
- Andreas Zeller, *Why Programs Fail*, and David Agans, *Debugging: The 9 Indispensable Rules* — the foundational texts on reading evidence and reasoning from symptom to cause.
