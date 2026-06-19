---
title: "Resource Leaks: File Descriptors, Sockets, and Connections That Run Out at 4pm"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Learn why a service that ran clean for weeks suddenly throws Too many open files at peak load, how to count and classify open descriptors with lsof and ss before you touch the code, and the structural fix that flatlines a climbing FD count to zero on every path including the error path."
tags:
  [
    "debugging",
    "software-engineering",
    "file-descriptors",
    "resource-leaks",
    "connection-pool",
    "sockets",
    "lsof",
    "close-wait",
    "context-managers",
    "root-cause-analysis",
  ]
category: "software-development"
subcategory: "Debugging"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/resource-leaks-fds-sockets-and-connections-1.png"
---

The pager goes off at 4:07pm. Not at 3am this time, which is somehow worse, because everyone is awake and watching. The service has been green for nineteen days. Then, right as the afternoon traffic peaks, it starts throwing `OSError: [Errno 24] Too many open files`, the load balancer marks every instance unhealthy, and the orchestrator begins a restart loop that buys you ninety seconds of health before the next instance hits the wall. By 4:20 you have a full outage and a channel full of people asking "what changed?" The honest answer is: nothing changed today. The bug shipped nineteen days ago. It has been leaking, quietly and monotonically, every single afternoon, and today the afternoon was just busy enough that the slow climb finally crossed a line the kernel will not let you cross.

That line is `RLIMIT_NOFILE`, the per-process cap on open file descriptors, and the reason this class of bug is so vicious is that it is *invisible until it isn't*. A descriptor leak does not crash where the bug lives. It crashes far away, in some innocent `accept()` or `open()` or `connect()` that had the bad luck to be the call that asked for descriptor number 1025 when the limit was 1024. The stack trace points at the victim, not the culprit. The leak itself produces no error, no log line, no exception — it just consumes a slot in a finite table and never gives it back. You only find out the table is full when the next allocation fails, and by then the leaking code is long gone from any trace you can see.

![A layered diagram showing an application handle becoming a small integer file descriptor that indexes a per-process FD table capped by RLIMIT_NOFILE, where exhaustion makes open return EMFILE](/imgs/blogs/resource-leaks-fds-sockets-and-connections-1.png)

This post is about the whole family of "I ran out of a finite resource" bugs: leaked file descriptors and sockets, connection pools that drain and never refill, thread pools where every worker is parked on something slow, and the `CLOSE_WAIT` pileup that is a `close()` you forgot to call, multiplied by ten thousand. By the end you will be able to do five concrete things. Count and classify open descriptors on a live process with `lsof` and `/proc`. Watch the FD count under load and read a *monotonic* climb as the unambiguous signature of a leak. Use `ss -s` to bucket sockets by TCP state and read a growing `CLOSE_WAIT` count as "the application is not closing its end." Read connection-pool and thread-pool metrics to tell a leak apart from a stall that only *looks* like a deadlock. And apply the one structural fix that kills the entire family: release the resource on *every* path — including the error path — with `with`, `defer`, try-with-resources, or RAII, plus a bounded pool with an acquire timeout so a slow dependency fails fast instead of blocking forever.

We will stay on the series spine the whole way: **observe → reproduce → hypothesize → bisect → fix → prevent.** This bug class punishes guessing more than almost any other, because the symptom (a failed `open()`) and the cause (a missing `close()` somewhere else entirely) are separated by both code distance and time. If you want the philosophy behind that loop, start with the intro map, [stop guessing and use the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging). This post is the FD-and-pool chapter of it.

## 1. What a file descriptor actually is

Let's build the mechanism from the bottom, because every fix downstream only makes sense once you know exactly what is being leaked.

When your program calls `open("/etc/hosts")` or `socket()` or `pipe()`, the kernel returns a small non-negative integer — `3`, then `4`, then `5`, and so on. That integer is a **file descriptor** (FD). It is not the file. It is an *index into a per-process table* the kernel keeps for you, the file descriptor table. Entry `3` in that table points at a kernel `struct file` (the open-file description), which in turn points at the inode for a real file, or at a socket object, or at the read or write end of a pipe. Three FDs are pre-wired before your code runs: `0` is stdin, `1` is stdout, `2` is stderr. Your first `open()` almost always gets `3` because the kernel hands back the lowest available index.

Two facts about that table are the whole story. First, it is **per process**, not per thread and not global — every process has its own. Second, it is **bounded**. The bound is `RLIMIT_NOFILE`, the soft limit you see with `ulimit -n` from a shell. On a stock Linux box that is often `1024`; in containers it might be `1024`, `65536`, or something a base image set badly. When your process has that many entries in its FD table and asks for one more, the kernel refuses: `open()` returns `-1` with `errno == EMFILE` ("too many open files" for *this process*), and the closely related `ENFILE` means the *system-wide* open-file table is full. In Python you see `OSError: [Errno 24]`; in Java, `java.net.SocketException: Too many open files`; in Go, `accept tcp: too many open files`; in Node, `EMFILE`.

The mechanism that makes leaks possible is exactly this finiteness combined with the fact that **closing is your job**. The kernel will not garbage-collect an FD just because you stopped using the object in your language. In C, you call `open()`, you call `close()`. In a garbage-collected language the situation is sneakier: the *object* wrapping the FD might get collected eventually, and many runtimes register a finalizer that closes the FD when that happens — but finalization runs at the GC's whim, which can be *seconds to minutes* after the object became unreachable, or never if the process is shutting down or under memory pressure that delays GC. So "the GC will close it" is true in the way "the check is in the mail" is true. Under load, you allocate FDs far faster than a lazy finalizer reclaims them, and the table fills before the GC catches up. (If you want the memory-side mirror of this — when an *object* leaks rather than a *descriptor* — the sibling post on [hunting memory leaks and bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) is the companion piece; the diagnostic shape is the same, the resource is different.)

So a **resource leak** is precisely: you acquired a slot in a finite table — an FD, a pooled connection, a thread — and you never returned it. Do that on a hot path, even occasionally, and the count climbs until the table is full and the next acquirer falls over.

It's worth being precise about *which* limit you hit, because there are three and people confuse them constantly. The **soft limit** (`ulimit -Sn`, the one your process actually enforces) is what your code runs into; the **hard limit** (`ulimit -Hn`, the ceiling a process can raise its own soft limit to without privilege) is the maximum you can self-bump; and the **system-wide limit** (`/proc/sys/fs/file-max`, the kernel's global cap across all processes) is what `ENFILE` means when you hit it. Almost every "too many open files" you'll debug is the per-process soft limit (`EMFILE`), not the system-wide one (`ENFILE`) — a single leaking service rarely exhausts the whole machine, it exhausts *itself* first. You can read all three live:

```bash
$ ulimit -Sn                          # soft limit for this shell/process: 1024
1024
$ ulimit -Hn                          # hard limit you may raise the soft limit up to
524288
$ cat /proc/sys/fs/file-max           # system-wide ceiling across ALL processes
9223372
$ cat /proc/4823/limits | grep "open files"
Max open files            1024                 1024                 files
```

That last command — reading `/proc/PID/limits` — is the one to remember, because it tells you the *running* process's actual limit, which can differ from your shell's `ulimit` (the service might have been started by systemd with a `LimitNOFILE=` directive, or by a container runtime with a different default). I have watched an hour evaporate because someone checked `ulimit -n` in their SSH session (65536) and concluded "we have plenty of headroom," while the actual service, started by systemd with `LimitNOFILE=1024`, was dying at 1024. Always read the limit *of the process*, not of your shell.

There's one more piece of mechanism that explains the timing of the crash. Recall the kernel hands back the *lowest available* FD number. So as you leak, the numbers climb: `3, 4, 5, ..., 1021, 1022, 1023`, and the *next* request for a descriptor — number `1024`, the one that would exceed the count the limit allows — fails. This is why the crash lands on whatever call happens to be next: an inbound `accept()` (you can't take new connections), an outbound `connect()` (you can't reach a dependency), an `open()` of a config or log file (you can't even write the error log explaining why you're failing — a particularly cruel variant where the leak silences your own diagnostics). The leak is indiscriminate about *who* it kills; it kills the next allocator, and the next allocator is whoever was unlucky.

Now let's see where the missing release usually hides.

## 2. The leak lives on the error path

Here is the single most important sentence in this post: **resource leaks almost always live on the error path, not the happy path.** Engineers test the happy path. The happy path closes its FD. The error path is the one nobody ran in the test that mattered, and the error path is where the `close()` got skipped.

Consider this deliberately ordinary Python HTTP client helper:

```python
import socket

def fetch_status(host, port, path):
    s = socket.create_connection((host, port), timeout=5)
    request = f"GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n"
    s.sendall(request.encode())
    response = s.recv(65536)            # <-- can raise socket.timeout
    status_line = response.split(b"\r\n")[0]
    code = int(status_line.split(b" ")[1])   # <-- can raise ValueError/IndexError
    s.close()                          # <-- only runs if nothing above threw
    return code
```

Read it the way the bug reads it. `create_connection` succeeds and you now hold a socket — one FD. Then `recv` raises `socket.timeout` because the upstream is slow. Control jumps straight out of the function. `s.close()` never runs. The socket FD is now leaked: the Python object `s` becomes unreachable when the exception unwinds the frame, and *eventually* a finalizer might close it, but until then that descriptor sits in your table doing nothing. Or `recv` returns garbage, `status_line.split(b" ")[1]` raises `IndexError`, and again `close()` is skipped. Every slow upstream, every malformed response, every timeout is a leaked FD. On a healthy day with few errors you leak slowly. On a bad-upstream day you leak fast — and a bad-upstream day is exactly a busy afternoon when the dependency is also under load.

![A before and after figure contrasting a manual close that leaks a socket on the exception path against a context manager that runs close on every path and keeps the descriptor count flat](/imgs/blogs/resource-leaks-fds-sockets-and-connections-2.png)

The control-flow picture is worth making explicit, because it generalizes to every language. You acquire the resource. You do some work. The work either succeeds (and reaches your `close()`) or throws (and jumps *past* your `close()`). A `close()` placed *after* the work covers the success path and leaks on the throw path. The only placement that covers *both* exits is one that runs during unwinding: a `finally` block, a `defer`, a `with` context manager, or a stack-allocated RAII object whose destructor fires as the scope exits.

![A control-flow graph showing open and work forking into a success branch that closes and an exception branch that jumps past close and leaks, with a finally or defer node that releases the descriptor on both branches](/imgs/blogs/resource-leaks-fds-sockets-and-connections-3.png)

Here is the same helper, fixed, in four languages. The fix is structural — it changes *where* the close lives, not whether you remembered to call it.

```python
import socket
from contextlib import closing

def fetch_status(host, port, path):
    with closing(socket.create_connection((host, port), timeout=5)) as s:
        s.sendall(f"GET {path} HTTP/1.0\r\nHost: {host}\r\n\r\n".encode())
        response = s.recv(65536)
        return int(response.split(b"\r\n")[0].split(b" ")[1])
    # close() is guaranteed here on EVERY exit, including exceptions
```

```go
func fetchStatus(host, port, path string) (int, error) {
    conn, err := net.DialTimeout("tcp", host+":"+port, 5*time.Second)
    if err != nil {
        return 0, err
    }
    defer conn.Close()                 // runs on every return, panic included
    fmt.Fprintf(conn, "GET %s HTTP/1.0\r\nHost: %s\r\n\r\n", path, host)
    buf := make([]byte, 65536)
    n, err := conn.Read(buf)           // even if this errors, defer still closes
    if err != nil {
        return 0, err
    }
    // parse buf[:n] ...
    return parseCode(buf[:n])
}
```

```java
// try-with-resources: close() is called on the Socket on every exit
int fetchStatus(String host, int port, String path) throws IOException {
    try (Socket s = new Socket()) {
        s.connect(new InetSocketAddress(host, port), 5000);
        s.getOutputStream().write(
            ("GET " + path + " HTTP/1.0\r\nHost: " + host + "\r\n\r\n").getBytes());
        byte[] buf = new byte[65536];
        int n = s.getInputStream().read(buf);   // throws? s.close() still runs
        return parseCode(buf, n);
    }
}
```

```cpp
// RAII: the destructor of Socket calls close() when fd goes out of scope
struct Socket {
    int fd;
    explicit Socket(int f) : fd(f) {}
    ~Socket() { if (fd >= 0) ::close(fd); }   // runs during stack unwinding too
    Socket(const Socket&) = delete;           // no accidental double-close
};

int fetch_status(const char* host, int port) {
    Socket s(connect_to(host, port));         // throws below? ~Socket() still fires
    send_request(s.fd);
    return parse_code(read_all(s.fd));        // exception unwinds -> close() runs
}
```

Same idea everywhere: bind the resource's lifetime to a scope and let the language's unwinding machinery guarantee release. Python's `with`, Go's `defer`, Java's try-with-resources, and C++ RAII are four spellings of one rule — *release on every path*. The reason this is a *fix* and not a *patch* is that it is impossible to forget on the error path, because there is no separate error path to forget: the close is attached to the scope, and the scope always exits.

Node.js deserves its own mention because its event-loop model produces a subtler version of the leak. There is no `with` or `defer`, and the close is asynchronous, so the idiom is `try/finally` around the await, or explicit cleanup in every branch:

```js
async function fetchStatus(host, port, path) {
  const socket = net.connect(port, host);
  try {
    await once(socket, 'connect');
    socket.write(`GET ${path} HTTP/1.0\r\nHost: ${host}\r\n\r\n`);
    const [data] = await once(socket, 'data');   // rejects on error/timeout
    return parseCode(data);
  } finally {
    socket.destroy();    // runs whether the await resolved OR rejected
  }
}
```

The Node trap is that an *unhandled* `'error'` event on a socket that you never attached an error handler to will crash the whole process — but a socket you forgot to `destroy()` on a *handled* error path just leaks silently, and because Node is single-process and often runs for days, the leak accumulates exactly like the others. The same `lsof`/`/proc` diagnostics apply unchanged; Node's FDs are kernel FDs like everyone else's.

The table below is how the four languages spell the one rule, with the specific gotcha each one hides:

| Language | Release idiom | The gotcha that still leaks |
| --- | --- | --- |
| Python | `with` / `contextlib.closing` | A bare `open()` not wrapped; a generator that never finishes (its `finally` never runs) |
| Go | `defer x.Close()` | `defer` in a *loop* runs at function end, not iteration end — defers pile up |
| Java | try-with-resources | A resource constructed *outside* the `try (...)` header isn't auto-closed |
| C++ | RAII (destructor) | A raw FD held in a struct without a destructor; a moved-from object double-closing |
| Node.js | `try { } finally { destroy() }` | Forgetting an `'error'` handler (crash) or a `destroy()` (silent leak) |

That Go loop gotcha is worth a beat because it bites everyone once: `defer` schedules the call for *function* return, not for the end of the current loop iteration. So `for _, f := range files { fd, _ := os.Open(f); defer fd.Close(); process(fd) }` opens every file and closes *none* of them until the function returns — if the loop has 10,000 iterations, you hold 10,000 FDs simultaneously and may hit the limit mid-loop. The fix is to put the body in a closure (so the `defer` fires per iteration) or to `Close()` explicitly at the end of each iteration. The idiom is right; the *scope* was wrong.

## 3. Observe: count and classify the descriptors

Before we fix anything in anger, we observe. The cardinal rule of this bug class: **never guess which resource is leaking — count it.** You have three cheap instruments, and they answer three different questions.

The first instrument is `lsof` ("list open files"), which on a live process gives you not just a count but a *classification* — files versus sockets versus pipes — and that classification is half the diagnosis. Find your process ID and run:

```bash
# How many FDs does PID 4823 hold right now?
$ lsof -p 4823 | wc -l
1019

# Classify them: what KIND of descriptor is leaking?
$ lsof -p 4823 | awk '{print $5}' | sort | uniq -c | sort -rn
    842 IPv4        # 842 sockets  <-- this is where the leak is
     91 REG         # 91 regular files
     47 DIR
     21 CHR
     12 unix
      6 pipe
```

That `842 IPv4` is the smoking gun: the leak is *sockets*, not files. If instead you saw `REG` (regular files) climbing, you would be hunting unclosed file streams; if `pipe`, leaked subprocess pipes. `lsof` told you the category before you read a single line of code, which lets you ignore three-quarters of the codebase. The deeper companion technique — watching the actual syscalls that open and never close these FDs — is the subject of the sibling post on [seeing what a process really does with syscall tracing](/blog/software-development/debugging/seeing-what-a-process-really-does-syscall-tracing); `strace -f -e trace=openat,socket,close` will literally show you the unbalanced opens.

The second instrument is `/proc`, which is faster than `lsof` and safe to poll in a tight loop — important when you are watching the count *move* under load:

```bash
# Instantaneous FD count for PID 4823 (the kernel's own view)
$ ls /proc/4823/fd | wc -l
1019

# Watch it climb under load, once per second
$ while true; do
    echo "$(date +%T)  $(ls /proc/4823/fd | wc -l)"
    sleep 1
  done
14:02:01  118
14:02:02  118
14:02:03  124
14:02:04  131    # climbing under traffic...
14:02:05  138
14:02:06  145    # +7/sec, and it never comes back down
```

That is the *signature*. A healthy process's FD count is a sawtooth — it rises as requests arrive and falls as they finish, oscillating around a steady mean. A leaking process's FD count is **monotonic**: it climbs and never returns to baseline. If you watch for sixty seconds and the floor of the count keeps rising, you have a leak, full stop. This is the single most reliable observation in the whole investigation, and it requires no debugger, no code reading, and no theory — just `ls /proc/PID/fd | wc -l` in a loop. (To make the climb obvious, generate load yourself with a tool like `wrk`, `hey`, or even a `for` loop of `curl` calls while you watch the count.)

![A diagnostic matrix mapping lsof, proc FD listing, ss socket-state buckets, and pool metrics each to what they show and the specific leak signal that counter produces](/imgs/blogs/resource-leaks-fds-sockets-and-connections-5.png)

There is a fourth instrument worth knowing, for when you want to catch the leak *in the act of being created* rather than after the fact: tracing the `open`/`socket`/`close` syscalls directly. `lsof` and `/proc` show you the *standing inventory* of FDs; `strace` shows you the *flow* — every acquire and every release as it happens — so an *imbalance* between opens and closes is the leak, made visible at the syscall boundary where the kernel actually hands out and reclaims the descriptors:

```bash
# Trace open/socket/close on a running process, follow child threads (-f),
# and timestamp each line. Watch opens-without-matching-closes accumulate.
$ strace -f -p 4823 -e trace=openat,socket,accept,close -tt 2>&1 | head -20
14:22:01.114  socket(AF_INET, SOCK_STREAM, 0) = 7      # acquired FD 7
14:22:01.118  connect(7, {sa_family=AF_INET, ...})
14:22:01.402  recvfrom(7, ...)                          # timeout/error here...
14:22:01.402  # ...and NO close(7) ever follows. FD 7 leaked.
14:22:01.511  socket(AF_INET, SOCK_STREAM, 0) = 8      # next request, FD 8...
14:22:01.802  # ...also no close(8). Each request leaks exactly one socket.
```

A tidier way to *quantify* the imbalance over a window is to count opens versus closes in a sample:

```bash
# Sample 10 seconds of syscalls and tally acquire vs release
$ timeout 10 strace -f -p 4823 -e trace=socket,close -c 2>&1 | grep -E 'socket|close'
% time   calls   syscall
 51.2     1240   socket     # 1240 sockets opened in 10s
 12.8      610   close      # only 610 closed -> ~630 leaked in 10s = +63/sec
```

Opens far exceeding closes over a steady window is the leak, quantified, at the exact layer the kernel cares about. A caution: `strace` is *intrusive* — it stops the process at every traced syscall via `ptrace`, which can slow a hot process by 10–100×, so use it on a canary or a reproducer, never on a healthy high-traffic prod instance you can't afford to slow. The lighter-weight, production-safe equivalent is `bpftrace`/eBPF, which traces the same syscalls with near-zero overhead by running in the kernel; the full treatment of both is in the [syscall-tracing sibling post](/blog/software-development/debugging/seeing-what-a-process-really-does-syscall-tracing). For a first look on a non-critical process, though, `strace -f -e trace=socket,close -c` is the fastest way to *see* the acquire/release imbalance with your own eyes, which is often what finally makes the bug click.

The third standing instrument is `ss` (or older `netstat`), which buckets sockets by their TCP connection state — and that bucketing is how you tell *why* the sockets are leaking, which we get to next.

#### Worked example: the afternoon that died, traced in eleven minutes

A payments-adjacent service threw `Too many open files` every afternoon between 3:45 and 4:30, then recovered after the nightly traffic dip, so it looked transient and nobody had caught it in the act. We caught it. At 3:30, with traffic ramping, we grabbed the PID and started the `/proc` loop. Baseline was 140 FDs. By 3:40 it was 410. By 3:50, 720, climbing about **+200 per minute** during the busy stretch. Monotonic — never a single dip. So: a leak, and `lsof -p PID | awk '{print $5}' | sort | uniq -c` said `IPv4` was 88% of it. Sockets. Then `ss -tan | awk '{print $1}' | sort | uniq -c` showed the killer: `CLOSE_WAIT` had gone from a handful to over six hundred, and it tracked the FD count almost exactly. `CLOSE_WAIT` means *the peer closed and we never did* — a missing `close()` on our side. We grepped for the outbound HTTP client used by the daily-report endpoint (which only gets heavy in the afternoon), found a `requests.get(...)` whose `.close()` was inside a branch that the timeout path skipped, wrapped the call in a `with` block, and shipped it. Next afternoon: FD count peaked at 190 and oscillated. Flat. The whole diagnosis was eleven minutes of `lsof`, `/proc`, and `ss`, and *zero* time in a debugger. The numbers are illustrative of the shape — your exact rates will differ — but the method is exactly this and it is fast.

## 4. Sockets have states, and one of them is your leak

A file FD is either open or closed. A *socket* FD is more interesting: a TCP connection moves through a defined set of states, and two of those states are where socket leaks announce themselves. Understanding them turns `ss` output from noise into a diagnosis.

When you finish with a TCP connection, the close is a four-way handshake, not an instant. Whoever calls `close()` first sends a `FIN`. The two states that matter for debugging are:

- **`CLOSE_WAIT`** means *the remote peer sent us a `FIN` (they closed their end) and our application has not yet called `close()`.* The connection is sitting half-closed, waiting for *us* to close, and it will sit there forever if we never do. **A pile of `CLOSE_WAIT` is, with near-certainty, a bug in *your* code: a missing or skipped `close()`.** This is the socket-leak signature.
- **`TIME_WAIT`** means *we closed first*, and the socket is in a mandatory cool-down (typically ~60s, twice the maximum segment lifetime) so that any late, duplicate packets from the old connection don't get misdelivered to a new one reusing the same port pair. A pile of `TIME_WAIT` is usually *not* a bug — it is normal for a busy client making many short-lived connections. It can exhaust *ephemeral ports* under extreme connection churn, but it is the kernel doing the right thing, not your code leaking.

So the diagnostic discipline is: **`CLOSE_WAIT` growth is your leak; `TIME_WAIT` growth is usually load.** Confusing the two sends you down the wrong path, so here is how to read them with `ss`:

```bash
# Summary of all sockets by state (the fastest first look)
$ ss -s
Total: 1041
TCP:   1024 (estab 41, closed 12, orphaned 0, timewait 12)

# Bucket sockets by TCP state and count them
$ ss -tan | awk 'NR>1 {print $1}' | sort | uniq -c | sort -rn
    612 CLOSE-WAIT     # <-- 612 connections WE never closed. This is the leak.
    180 ESTAB
     41 TIME-WAIT
      9 LISTEN

# Watch CLOSE-WAIT climb under load (the leak in motion)
$ while true; do
    printf '%s  CLOSE_WAIT=%d\n' "$(date +%T)" \
      "$(ss -tan state close-wait | grep -c '^')"
    sleep 2
  done
14:11:02  CLOSE_WAIT=188
14:11:04  CLOSE_WAIT=201
14:11:06  CLOSE_WAIT=214    # +6.5/sec, monotonic -> a close() leak
```

The comparison below is the cheat sheet I keep in my head when I run `ss` against a sick process. The whole game is mapping the state bucket that is growing to the action you take.

| Socket state | What it means | Growing count signals | Your move |
| --- | --- | --- | --- |
| `CLOSE_WAIT` | Peer closed; *you* haven't | A missing `close()` in your code (leak) | Find the unclosed socket, add `with`/`defer`/finally |
| `TIME_WAIT` | *You* closed; OS cool-down | High connection churn (usually normal) | Reuse connections (keep-alive/pool); rarely a code bug |
| `ESTAB` | Live connection | Real concurrency, or held-open leaks | Check if count matches real in-flight requests |
| `FIN_WAIT_2` | You closed; peer hasn't FIN'd | Peer is slow/buggy to close | Set a timeout; not usually your leak |
| `SYN_SENT` (many) | Connects not completing | Upstream down or firewalled | Network/dependency issue, not an FD leak |

The reason `CLOSE_WAIT` is such a clean signal is the mechanism: the only way out of `CLOSE_WAIT` is for *your* process to call `close()` on that socket. The kernel cannot do it for you, because as far as TCP is concerned you might still want to *send* data on a half-closed connection (the peer closed *their* sending direction, not yours). So a stuck `CLOSE_WAIT` is definitionally a `close()` your application owed and didn't pay. Ten thousand of them is the same one-line bug, ten thousand times. If you want to confirm *which* socket and *which* peer, `lsof -p PID -i` lists every socket FD with its local and remote address, and you can match the remote address to the dependency you forgot to close.

## 5. Reproduce it before you fix it

You have observed a leak in prod. The series rule is absolute: **reproduce it before you trust your fix.** A leak you cannot reproduce is a leak you cannot prove you fixed — you will ship a change, watch the afternoon, and *believe* it worked, but "the count didn't climb today" might just mean today was quiet. Reproduce it on demand and the proof is unambiguous: the count climbs before your change and is flat after, on the same load.

For an FD leak the reproducer is wonderfully simple, because the leak is deterministic per error: drive the error path in a loop and watch `/proc`. Here is a self-contained reproducer that *manufactures* the leak so you can see the mechanism, then watch it die:

```python
import socket, os, time

def leaky_call():
    # Acquire a socket, then deliberately hit an error path that skips close()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.connect(("example.com", 80))
    raise RuntimeError("boom on the error path")  # close() never reached
    s.close()                                     # dead code, never runs

def fd_count():
    return len(os.listdir(f"/proc/{os.getpid()}/fd"))

for i in range(1, 2001):
    try:
        leaky_call()
    except RuntimeError:
        pass
    if i % 200 == 0:
        print(f"after {i} calls: {fd_count()} FDs open")
        # after 200 calls: 205 FDs open
        # after 400 calls: 405 FDs open   <- +1 FD per call, dead linear climb
        # after 600 calls: 605 FDs open
        # ... eventually: OSError: [Errno 24] Too many open files
```

Run that and you watch the FD count rise by exactly one per call until it hits the `ulimit -n` cap and the program dies with the *same* `Errno 24` you saw in prod — a faithful local reproduction. Now apply the fix (`with closing(socket.socket(...)) as s:`) and re-run: the count flatlines at the baseline, 2000 calls complete, no `OSError`. That contrast — **2000 calls, climbs to crash before / flat after** — is your proof, and it took thirty seconds to produce.

#### Worked example: proving the fix with a 10,000-iteration loop

The cleanest proof this bug class offers is a counted loop with the FD gauge printed at the end. We had a fix in review and a reviewer (correctly) refusing to merge it on the strength of "looks right." So we wrote the proof. The harness drove the *actual* leaking endpoint through a deliberately-failing upstream (a local `nc -l 9999` that accepted connections and replied nothing, forcing the recv timeout that triggered the leak path), 10,000 times, measuring `ls /proc/PID/fd | wc -l` before and after. On the unpatched build: start 142 FDs, end 8,341 FDs — a near-perfect +0.82 FDs per iteration (some completed before timeout), and the run actually aborted at iteration 8,051 with `Errno 24` once it hit the 8,192 cap, which was itself a clean confirmation of the mechanism. On the patched build with the `with` block: start 141 FDs, end 149 FDs over the same 10,000 iterations — flat, the small wobble just being the handful of genuinely-in-flight sockets at the moment of measurement. That's the whole proof: **+0.82/iter to crash, before; flat at ~145, after**, same load, same upstream, same endpoint. A reviewer cannot argue with a flatline. The numbers here are the real shape of such a run; your absolute values depend on your `ulimit` and timeout, but the *before climbs linearly to the cap and after is flat* contrast is exactly what you'll see, and it's the only proof that distinguishes "fixed" from "got lucky on a quiet day."

For the prod-shaped version where you cannot edit the service to add a `raise`, you reproduce by *generating the error condition*. If the leak is on the timeout path, point the client at a deliberately slow or dead endpoint (a `nc -l` that accepts and never replies, or a `toxiproxy`/`iptables` rule that drops the response) and pour load through it while watching `ss -tan state close-wait | wc -l`. The `CLOSE_WAIT` count climbing under your synthetic bad-upstream is the reproduction. The deeper treatment of *how* to reliably reproduce intermittent and load-dependent bugs is its own discipline — the sibling [reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) goes into the repeat-until-fail loops and fault injection that make a flaky leak deterministic.

## 6. The timeline: why it always breaks at peak

Step back and look at the *shape over time*, because it explains the single most confusing thing about this bug: why it breaks at 4pm and not at 4am, and why "nothing changed today."

![A timeline showing the file descriptor count climbing monotonically through the afternoon from an idle baseline past a near-cap warning until peak load hits the EMFILE wall and the service enters a restart loop](/imgs/blogs/resource-leaks-fds-sockets-and-connections-4.png)

The leak rate is proportional to the *error rate*, and the error rate is proportional to *load* (busy upstreams time out more; malformed responses arrive more; the slow path fires more often). So the FD count is an integral: it accumulates all day, faster when busy. In the morning the floor creeps up slowly. Through the day it ratchets higher with each traffic bump, and crucially **it never falls back to baseline** because nothing is closing those leaked FDs. By mid-afternoon you are at 980 of 1024, riding just under the cap, with no alarm because nothing is *broken* yet. Then the daily peak adds the last few dozen connections, `accept()` asks for descriptor 1025, the kernel says `EMFILE`, and the service falls over — *exactly* at peak, every day, with no deploy to blame.

This is why the bug is so good at hiding. A leak that takes hours to cross the cap is invisible to every test (which runs for seconds), invisible to the morning, and invisible to any single request (each one leaks *one* FD, harmless in isolation). It is only visible as an *aggregate trend over time under load*, which is precisely the thing humans are worst at noticing and dashboards are best at. The prevention follows directly: **alert on the trend, not the failure.** A simple alert on "open FD count > 70% of `RLIMIT_NOFILE`" or "FD count has risen monotonically for 30 minutes" catches the leak at 2pm with two hours of runway, instead of paging you mid-outage at 4:07.

#### Worked example: the +200/min climb that flatlined

Let me put real numbers on the integral. The service had `ulimit -n` of 8192 (someone had bumped it once during a previous fire, which bought time but didn't fix anything — bumping the limit is a snooze button, not a cure). Idle baseline was 140 FDs. The afternoon error rate drove the leak at roughly **+200 FDs/min** during the busy 3–5pm window. From 140, at +200/min, you hit 8192 in about `(8192 - 140) / 200 ≈ 40` minutes of sustained peak — which is exactly why it died around 4:05 after the peak began around 3:25. We added the `with` block, redeployed, and re-ran the synthetic-load reproducer first to confirm the count was flat before trusting prod. In the reproducer: before, 5000 error-path calls drove FDs from 140 to 5140; after, 5000 calls held at 142–151 (just the live concurrent ones). In prod the next afternoon, the peak FD count was 230 and oscillating. The MTTR math is worth saying out loud: a trend alert at 70% of 8192 (about 5700 FDs) would have fired at roughly 3:53, eleven minutes before the outage — enough to fail over or restart on *our* schedule instead of the kernel's.

## 7. Pools: the leak that looks like a deadlock

Everything so far has been about a *raw* finite resource — FDs in the kernel table. But the same pattern repeats one level up, in the **pools** your application builds *on top of* FDs, and here the leak wears a disguise: it looks like a hang or a deadlock, not a "too many open files" error.

A **connection pool** is a fixed-size set of pre-opened connections (to a database, a cache, a downstream service) that requests *borrow* and *return*. The pool exists because opening a connection is expensive — a TCP handshake plus a TLS handshake plus an auth round-trip — so you pay that cost once per connection and reuse it across thousands of requests. The pool has a hard maximum, say 20 connections. (The mechanics, sizing, and trade-offs of pools are covered in depth in [database connection pooling](/blog/software-development/database/database-connection-pooling); here we care only about how they *fail*.) The critical contract: every request that *checks out* a connection must *check it back in*, or that connection is gone from the pool forever — a **pooled-connection leak**, structurally identical to the FD leak, just at a higher layer.

![A flow graph showing incoming requests entering a fixed-size DB pool where a slow query holds all twenty connections checked out, forcing new requests into an unbounded wait that stalls every thread and mimics a deadlock](/imgs/blogs/resource-leaks-fds-sockets-and-connections-6.png)

There are two failure modes and they feel identical from the outside, so you have to distinguish them with metrics:

**Pool leak.** A request checks out a connection and, on some error path, never returns it (the same missing-`close()`/missing-`return-to-pool` bug). Each leaked connection shrinks the usable pool. After 20 such errors the pool is empty *permanently*, even when the database is healthy and idle. Every subsequent request blocks waiting for a connection that will never come back. The service stalls. The database shows almost no activity — which is the tell, and the trap: ops looks at a quiet database and says "the DB is fine," when the problem is that your app *can't reach* the healthy DB because it leaked away all its tickets to it.

**Pool starvation (no leak).** Every connection is checked out *legitimately* — by a request that is currently blocked on a *slow* operation (an un-indexed query taking 8 seconds, say). The connections aren't leaked; they're just all *busy* at once because each one is held for the full duration of a slow query. With 20 connections and queries taking 8s, you can only retire 20 queries per 8s = 2.5 queries/sec; if requests arrive at 50/sec, the queue grows without bound and every new request waits. The database here is *very* busy (running 20 slow queries) — the opposite database signature from a leak.

Both modes present as "the service hangs and requests time out," which is why people reach for the word *deadlock*. It usually isn't a deadlock in the strict sense (a cycle of locks where each thread waits for one the other holds — the subject of the planned sibling post on deadlocks, livelocks, and starvation, `deadlocks-livelocks-and-starvation`). It's resource exhaustion: every thread is parked waiting for a connection, and there is no cycle, just an empty pool and an unbounded wait. The fix is different from a real deadlock's, so the distinction matters.

## 8. Method: read the pool metrics and the thread dump

You cannot `lsof` your way to a pool diagnosis, because a pooled connection that's checked out and a pooled connection that's leaked look *identical* at the FD layer — both are open sockets to the database. The distinguishing information lives in the **pool's own metrics** and in a **thread dump**. So this is where you instrument the pool and read what the threads are actually doing.

Every serious pool exposes its internal counters. Read them first, because they answer "leak or starvation?" in one line:

```python
# SQLAlchemy / HikariCP-style pool metrics, sampled once per second
# active   = connections currently checked out
# idle     = connections available in the pool
# waiting  = threads blocked waiting to check one out
# total    = active + idle (should equal pool max when busy)

13:40:01  active=20  idle=0  waiting=47   total=20    # pool is fully checked out, 47 waiting
13:40:02  active=20  idle=0  waiting=63   total=20    # waiting grows -> arrivals > completions
13:40:03  active=20  idle=0  waiting=81   total=20
```

`active=20, idle=0, waiting climbing` tells you the pool is drained and requests are queuing. Now you need the *second* question: are those 20 connections *working* (starvation) or *leaked* (leak)? That's where the thread dump comes in. Take a dump of all threads and look at what the ones holding connections are *doing*:

```bash
# Java: dump every thread's stack; grep for what the pool-holding threads do
$ jstack 4823 > dump.txt
$ grep -A 8 'state=RUNNABLE' dump.txt | grep -c 'socketRead0'
20    # all 20 connection-holders are blocked in socketRead0 -> waiting on the DB
```

```bash
# Python: py-spy dump shows every thread's live stack without stopping the process
$ py-spy dump --pid 4823
Thread 0x7f3c (active): "worker-1"
    _recv (socket.py)              # <-- blocked reading from the DB socket
    execute (psycopg2/...)
    run_query (app/reports.py:88)  # all 20 workers are HERE, on the same slow query
```

```bash
# Go: SIGQUIT prints all goroutine stacks; or use pprof's goroutine profile
$ kill -SIGQUIT 4823    # prints goroutine dump to stderr
# goroutine 412 [IO wait]:  net.(*conn).Read ...  database/sql.(*DB).query ...
# 20 goroutines all parked in conn.Read on the same query -> starvation, not leak
```

If the thread dump shows all 20 holders **actively blocked inside a query** (`socketRead0`, `_recv`, `conn.Read`), it's **starvation**: the connections are busy on a slow operation. The database is doing real work; find and fix the slow query (add the index — see [why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod)) and add an acquire timeout so the pool fails fast instead of queuing forever. If instead the thread dump shows the 20 holders are **doing something else entirely** — or there are no 20 holders at all, the connections are just *gone* with no thread holding them — it's a **leak**: a checkout that was never returned. For that, the modern pools have a built-in detector.

HikariCP's `leakDetectionThreshold` is the canonical example: set it to, say, 5000 (ms) and the pool will log a full stack trace of *any* connection that has been checked out longer than 5 seconds, with the message "Apparent connection leak detected" and the exact stack where it was borrowed. That stack trace points straight at the borrow site whose return path is broken. It is the single best tool for a *pooled*-connection leak, the way `lsof` is for a raw FD leak.

```java
// HikariCP: turn on leak detection + a hard acquire timeout
HikariConfig cfg = new HikariConfig();
cfg.setMaximumPoolSize(20);
cfg.setConnectionTimeout(2000);        // acquire() fails after 2s, NOT forever
cfg.setLeakDetectionThreshold(5000);   // log a stack for any conn held > 5s
// Now a leaked connection produces a logged stack trace at the borrow site,
// and an exhausted pool produces a fast SQLTimeoutException instead of a hang.
```

The comparison below is how I decide, at the dump, which mode I'm in. Same symptom, opposite database signature, opposite fix.

| | Pool leak | Pool starvation |
| --- | --- | --- |
| `active` / `idle` | active=max, idle=0, *permanently* | active=max, idle=0, *under load* |
| Database load | Near zero (DB is idle) | Very high (DB runs N slow queries) |
| Thread dump of holders | Not blocked in a query (or no holder) | All blocked in `recv`/`socketRead0` |
| Leak detector | Fires: "connection held > Ns" | Does not fire (work is legitimate) |
| Recovers on its own? | No — leaked forever | Yes — drains when slow op finishes |
| Root cause | Missing return-to-pool on error path | Slow operation + too-small pool |
| Fix | `with`/`try-with-resources`, leak detector | Fix the slow op + acquire timeout + size pool |

## 9. Thread-pool starvation: when every worker is parked

There is a third finite resource in this family, and it's the one that takes down the *whole* service rather than one endpoint: the **thread pool** (or its async cousin, the event loop). A web server typically has a bounded pool of worker threads — say 200 — and each incoming request occupies one worker for its entire duration. The pool exists for the same reason the connection pool does: threads are expensive (each needs a stack, a kernel scheduling entity), so you reuse a fixed set rather than spawning one per request. And it fails the same way: if every worker is *parked* on something slow, there are no workers left to handle *any* request, including the health check, so the entire instance goes dark.

The mechanism is a cascade, and it's why one slow dependency can take down a service that doesn't even seem to depend on it. Picture the 200-thread pool. A downstream service slows from 50ms to 5s. Requests that touch that downstream now hold their worker thread for 5 seconds instead of 50ms — a 100× longer hold. At any given moment, far more workers are tied up waiting. If enough traffic routes through the slow path, all 200 workers end up blocked on it simultaneously. Now a *completely unrelated* request arrives — one that doesn't touch the slow downstream at all — and there's no worker free to run it. It queues. The health check queues. The load balancer marks the instance unhealthy and routes traffic to the *other* instances, which were already absorbing the same slow downstream, and now also tip over. The slow dependency, through the shared thread pool, has converted itself into a total outage of a service that uses it for only 10% of requests.

The diagnosis is, once again, a thread dump — and it's the most visually obvious diagnosis in this whole post, because you'll see the *same stack* repeated dozens or hundreds of times:

```bash
# Java: count how many threads are stuck at the same frame
$ jstack 4823 | grep -A 30 '"http-worker' | grep 'at ' | sort | uniq -c | sort -rn | head
    198 at java.net.SocketInputStream.socketRead0(Native Method)
    198 at com.example.SlowClient.callDownstream(SlowClient.java:64)
     12 at ...
# 198 of 200 workers are blocked in the SAME downstream call. That IS the starvation.
```

When 198 of 200 threads share a stack frame, you don't need a theory — the dump *is* the root cause, pointing at the exact line (`SlowClient.java:64`) where every worker is stuck. `async-profiler` in wall-clock mode gives you the same picture as a flame graph, with the dominant blocked frame as the widest tower. The fix is layered: a **timeout** on the downstream call (so a worker is held for at most, say, 1s rather than indefinitely), a **bulkhead** (a separate, smaller pool dedicated to the slow downstream, so it can starve *itself* but can't consume the main pool), and a **circuit breaker** (stop calling the downstream entirely once it's clearly failing, returning a fast error and freeing all those workers at once). The unifying principle is identical to the connection pool's: **bound the hold time and isolate the blast radius**, so one slow thing can't eat the whole shared resource.

The async/event-loop world has the exact same failure with a different face: instead of blocked threads, you get a *blocked event loop*. A single synchronous CPU-bound call or a forgotten `await` on a slow I/O operation stalls the one thread that runs all callbacks, and *every* concurrent request stalls behind it. The diagnostic is a different tool (`node --prof`, or the event-loop-lag metric, or `py-spy` on the async worker showing where it's parked), but the shape is the same: a finite concurrency resource, fully consumed by something slow, starving everything else.

## 10. The structural fix: bound everything, time out everything

We've found leaks two ways (raw FD via `lsof`/`ss`, pooled via metrics/dump). Now the durable fix, which is the same philosophy at both layers: **bind release to scope, bound every pool, and time out every acquire.** Three rules.

**Rule one: release on every path.** Section 2 covered this for raw FDs. For pooled connections it's the identical move — borrow inside a scope that returns on every exit:

```python
# SQLAlchemy: the connection is returned to the pool on EVERY exit
def run_report(engine, customer_id):
    with engine.connect() as conn:          # checkout
        result = conn.execute(
            text("SELECT total FROM orders WHERE customer = :c"),
            {"c": customer_id},
        )
        return result.scalar()
    # connection returned to pool here, even if execute() raised
```

```go
// Go database/sql: rows.Close() returns the connection to the pool.
// The classic leak is forgetting rows.Close() -> connection never returned.
rows, err := db.Query("SELECT total FROM orders WHERE customer = $1", id)
if err != nil {
    return 0, err
}
defer rows.Close()      // <-- THIS is the return-to-pool. Forget it = pool leak.
for rows.Next() {
    // ...
}
return total, rows.Err()
```

**Rule two: bound the pool and time out the acquire.** An *unbounded* wait is the thing that turns a slow dependency into a total outage. If `acquire()` blocks forever, then one slow query parks every worker indefinitely and the service is down. If `acquire()` instead has a 2-second timeout, then when the pool is drained the request *fails fast* with a clear error you can alert on, the worker is freed to do other work, and you have converted an invisible infinite stall into a visible, bounded, alertable error.

![A before and after figure contrasting an unbounded acquire that blocks indefinitely and parks the thread against a bounded pool with a two second acquire timeout that fails fast with an alert and recovers](/imgs/blogs/resource-leaks-fds-sockets-and-connections-7.png)

This is the deepest lesson of the post, and it generalizes far beyond pools: **fail fast beats block forever.** A bounded queue that rejects when full sheds load gracefully; an unbounded queue that accepts everything converts a slowdown into a death spiral, because the backlog grows without limit and the latency rises until every request times out anyway — only now you also have a memory problem from the queued work. Every pool, every queue, every retry buffer should have a *bound* and a *timeout*. The trade-off is honest: a timeout means some requests fail during a spike that an infinite wait would have *eventually* served. But "eventually," under load, means "after the load balancer already gave up and after the backlog ate your heap," so the eventual success is a fiction. Fast, explicit failure is almost always the better operational position.

**Rule three: make leaks loud.** Turn on the leak detector (HikariCP `leakDetectionThreshold`, or your pool's equivalent), add an FD-count gauge to your metrics, and alert on the *trend*. The goal is that the *next* leak you introduce is caught by a dashboard at 2pm with runway, not by a page at 4:07 mid-outage.

#### Worked example: the stall that was starvation, fixed in two moves

A reporting service "deadlocked" under afternoon load — every request hung, then 504'd at the load balancer's 30s timeout. Classic "it's a deadlock" panic. We pulled the pool metrics: `active=20, idle=0, waiting=140 and rising`. Pool drained. Then `py-spy dump` on the process: all 20 workers were parked in `_recv` inside the same `run_query` frame — *blocked in a query*, not leaked. So: **starvation, not leak, not deadlock.** The query was a `SELECT ... WHERE status = 'pending'` against a 40-million-row table with no index on `status`, doing a full scan that took 7–9 seconds each. With 20 connections at ~8s/query, throughput was ~2.5 queries/sec against ~45 req/sec of arrivals — the queue could only grow.

Two moves fixed it. First, `CREATE INDEX CONCURRENTLY ON orders (status)` took the query from 8s to 9ms — a 900× drop — which alone restored throughput to ~2000 queries/sec, far above arrival rate, so the pool stopped draining. Second, we set the acquire timeout to 2s and the connection timeout on the driver, so that *if* the pool ever drains again, requests fail in 2s with a logged `pool timeout` error and an alert, instead of hanging for 30s and looking like a deadlock. Result: p99 latency went from "30s timeout wall" to 40ms, and the "deadlock" never recurred — because it was never a deadlock. The numbers (8s→9ms, the 900× factor) are the kind of swing an index produces on a full-scan query; the exact figure depends on row count and cache state, but the order of magnitude is real and this is the single most common "it's a deadlock" that turns out to be a missing index plus an unbounded pool wait.

## 11. War stories: when running out of a resource took down something real

This bug class has a long and instructive rap sheet. A few real and realistic cases, because seeing the *scale* at which "I forgot to close it" plays out is the best motivation to wrap every acquire in a scope.

**The CLOSE_WAIT pileup as a recurring outage pattern.** This is less a single famous incident than a genre — search any large engineering org's postmortems and you find it. A service makes outbound HTTP calls with a client that, on some error or non-2xx path, fails to release the underlying connection. Under normal traffic the leak is slow and the daily restart (deploys, autoscaling) masks it. Then a quiet weekend with no deploys lets a single instance run for 72 hours straight, its `CLOSE_WAIT` count climbs past the FD limit, and it falls over — while every *other* instance, recently restarted, looks fine, which makes the on-call think it's a "bad host" rather than a systemic leak. The tell is always the same: `ss -tan` on the sick instance shows hundreds or thousands of `CLOSE_WAIT` to one downstream address. The fix is always the same: a context manager / `defer` around the client call. The lesson: **frequent restarts hide leaks**, so a leak that "only happens on long-lived instances" is a leak that happens *always* and is merely papered over by your deploy cadence.

**Thundering-herd retries draining a pool.** A downstream service slows down. Every caller's request to it now takes longer, holding its pooled connection longer, so the pool drains. The callers' requests start timing out, and — here's the amplifier — a naive retry policy *retries immediately*, opening *more* connections to the already-struggling downstream, which slows it further, which times out more requests, which triggers more retries. The pool, the FDs, and the downstream all exhaust together in a feedback loop. This is the *retry storm* or *thundering herd*, and it is why retries must have *exponential backoff with jitter* and a *circuit breaker* that stops calling a failing dependency entirely. An unbounded pool with no acquire timeout and immediate retries is the perfect kindling for it; bounded pools, acquire timeouts, backoff, and circuit breakers are the firebreak. The deeper treatment of how these outages cascade across services is in [debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale).

**The ephemeral-port exhaustion variant.** A batch job opens a fresh connection per item to a downstream, closes it correctly (so no `CLOSE_WAIT` leak), but does so *thousands of times per second*. Each close leaves a socket in `TIME_WAIT` for ~60 seconds, and the client side draws from a finite range of *ephemeral ports* (often ~28,000 usable). At high enough churn, all ephemeral ports are stuck in `TIME_WAIT` and the next `connect()` fails with "cannot assign requested address" (`EADDRNOTAVAIL`) — a resource exhaustion that is *not* a code leak (every close is correct) but a *design* problem: you're not reusing connections. The fix is a connection pool / HTTP keep-alive so you make a handful of long-lived connections instead of tens of thousands of short-lived ones. This is the case where `TIME_WAIT` growth, which I told you is usually benign, actually bites — and the diagnosis is exactly the same instruments (`ss -s`, `ss -tan | ... TIME-WAIT`), reading a different bucket.

**The unclosed Java stream that the GC almost saved.** A JVM service read small files in a hot loop with `new FileInputStream(path)` and forgot to close them, relying — without realizing it — on the finalizer to clean up. For a while it *worked*, because `FileInputStream` historically registered a finalizer that closed the FD when the object was collected. But finalizers run on a single finalizer thread at the GC's discretion, and under load the allocation rate of these streams outran the finalizer's reclaim rate. The FD count climbed faster than finalization could lower it, and the service hit `Too many open files` while the *heap* was nearly empty (the stream objects were tiny). The diagnosis was a thread dump (showing the finalizer thread saturated) plus `lsof` (showing hundreds of `REG` file FDs to the same directory). The fix was try-with-resources, which closes deterministically at scope exit instead of "whenever the GC gets around to it." The lesson that generalizes: **never rely on finalization to release a scarce resource.** Finalizers are a backstop for memory, not a release mechanism for FDs, sockets, or locks — they run too late and too unpredictably. (Modern Java even deprecated finalization for exactly this reason.) This is the GC-language version of the error-path leak: the close was "going to happen," just not soon enough.

**A historical note on the shape, not a specific vendor incident.** Many large, public postmortems over the years describe the same skeleton: a service runs fine for days, a code path that leaks an FD or a pooled connection on errors goes unnoticed because restarts mask it, traffic or an upstream slowdown raises the error rate, the leak accelerates, and the service falls over at peak with a stack trace pointing at an innocent `accept()` or `connect()`. The specific resource varies — file handles, sockets, pooled DB connections, ephemeral ports, threads — but the *debugging method* is invariant: find which finite-resource counter is climbing monotonically, classify it, trace the unbalanced acquire/release to its borrow site, and bind the release to scope. If you internalize one thing from the war stories, let it be that the *symptom* is wildly varied (a hang, a crash, a 503 storm, a "bad host") but the *root cause* and the *fix* are a small, knowable set.

The through-line in all of these: a finite resource (FDs, pooled connections, ephemeral ports, threads), a code path or design that consumes them faster than it returns them, and a collapse that happens *far* from the consuming code and *long* after it started — at peak, on the long-lived instance, under the retry storm. Memorize the shape and you'll recognize the next one in minutes instead of hours.

## 12. Stress-testing your diagnosis

A root cause you haven't stress-tested is a hypothesis you got lucky with. Before you close the ticket, push on the diagnosis from the angles this bug class loves to hide in.

**"What if it only reproduces under load?"** Almost always true for this class — a single request leaks one FD, invisible. You *need* load to see the aggregate. So your reproducer must generate concurrent traffic (`wrk`, `hey`, `ab`, or a `for` loop of `curl`) while you watch `/proc/PID/fd`. If you can only reproduce under load, that's not a failure of your method — it's a *confirmation* that you're looking at an aggregate-over-time leak.

**"What if it only happens on long-lived instances?"** Then your deploy cadence is hiding it (the war story above). To catch it, take one instance out of the deploy rotation and let it run for days while you graph its FD count, or simulate the lifetime by pouring a day's worth of error-path traffic through a test instance in an hour. A leak that "only happens after 6 hours in prod" will happen in 20 minutes if you drive the error path hard enough.

**"What if the database looks healthy?"** That's the *signature of a pool leak*, not evidence against it. A drained-by-leak pool means a quiet database (no one can reach it) — the quiet DB is the clue, not the alibi. Always check pool metrics (`active`/`idle`/`waiting`) before you trust "the DB is fine."

**"What if I can't attach a debugger in prod?"** Good — don't. You almost never need one for this class. `lsof`, `ls /proc/PID/fd`, `ss -s`, `py-spy dump`, `jstack`, and pool metrics are all *non-intrusive*: they read the process's state without stopping it. (Attaching `gdb` to a live payments process and accidentally pausing it is a great way to turn a leak into an outage.) The entire investigation in this post is observation-only; that's by design.

**"What if bumping `ulimit -n` made it go away?"** It didn't go away — you moved the wall further out and bought time. The count is still monotonic; it now takes longer to hit the higher cap. Bumping the limit is the correct *emergency* move during an active outage (it restores service while you find the real bug) but a *terrible* permanent fix (you'll hit the new limit on the next busy day, just later). Always pair a limit bump with "and now we find the leak."

**"What if it's intermittent — some afternoons it's fine?"** The leak rate tracks the *error* rate tracks the *load*. A light afternoon leaks slowly and might not cross the cap before the nightly dip resets it (via restart or traffic drop). That intermittency is not randomness; it's the integral not quite reaching the threshold on quiet days. The trend alert (FD count slope) catches it regardless of whether it crosses the cap that particular day.

![A decision tree that starts from a too many open files error and branches on whether lsof shows files, sockets, or flat FDs with stalled requests, routing each to its specific structural fix](/imgs/blogs/resource-leaks-fds-sockets-and-connections-8.png)

That decision tree is the whole diagnostic compressed: read which counter moves, and it routes you to the resource class and the fix. `lsof` says files → unclosed streams → `with`/try-with-resources. `lsof` says sockets and `ss` says `CLOSE_WAIT` → missing socket close → `defer`/`with`. FDs flat but requests stall → pool → metrics + dump to split leak from starvation → leak detector or acquire timeout. You almost never need anything past this tree.

## 13. How to reach for this (and when not to)

Every tool here has a cost and a right moment. Here's the decisive version of when to use what, and when to put the tool down.

**Reach for `ls /proc/PID/fd | wc -l` in a loop first, always.** It is the cheapest, fastest, most reliable signal in the entire investigation. Five seconds of watching it tells you "leak or not" with near-certainty (monotonic = leak). Do this before you read any code, form any hypothesis, or attach anything. It costs nothing and it's non-intrusive.

**Reach for `lsof -p PID` to classify, once you know there's a leak.** It's slower than `/proc` (it walks every FD and resolves names), so don't poll it in a tight loop, but its *classification* (file vs socket vs pipe) is worth a hundred lines of code reading because it eliminates whole categories of suspect.

**Reach for `ss -s` / `ss -tan` when the leak is sockets.** The `CLOSE_WAIT` vs `TIME_WAIT` distinction is the difference between "your code bug" and "your design/load," and getting it right saves you from fixing the wrong thing.

**Reach for pool metrics + a thread dump (`py-spy dump`/`jstack`/SIGQUIT) when requests *stall* but FDs are flat.** That's the pool signature, and you cannot diagnose it from FDs alone. The dump tells you leak vs starvation in one read.

**Reach for the leak detector (`leakDetectionThreshold`) proactively, in config, forever.** It's the cheapest insurance against the *next* pooled-connection leak — it turns a future silent leak into a logged stack trace at the borrow site.

Now the *don'ts*, which matter just as much:

- **Don't attach `gdb`/`lldb` to a live prod process for an FD leak.** You don't need it (the leak is visible from `/proc` and `lsof` without stopping anything), and pausing a production process — especially one handling payments or holding locks — risks turning a slow leak into an immediate outage. This is the one case where the heavier debugger is strictly worse than the lightweight observer. (When you *do* need an interactive debugger, the sibling [the debugger is a microscope, use it](/blog/software-development/debugging/the-debugger-is-a-microscope-use-it) covers doing it safely — but not on a leaking prod process.)
- **Don't just bump `ulimit -n` and call it fixed.** Bump it to survive the active outage, then immediately hunt the leak. A higher cap with a monotonic climb is a slower-motion version of the same crash.
- **Don't set an *unbounded* acquire wait "to be safe."** It's the opposite of safe — it converts a slow dependency into a total stall. A bounded acquire with a timeout is the safe default; the failure it produces is fast and visible, which is what you want.
- **Don't add a `try/except: pass` around the leaking call to "stop the crashes."** Swallowing the error makes the leak *worse* (you keep going on the error path that leaks) and hides the signal. The fix is to release the resource, not to silence the exception.
- **Don't chase a leak by reading code first.** Observe first. The code is large; the `/proc` and `lsof` signals point you at the *kind* of resource and often the *peer address*, narrowing the code search by an order of magnitude before you open an editor.

## Key takeaways

- A **file descriptor is a small integer indexing a finite, per-process kernel table** capped by `RLIMIT_NOFILE` (`ulimit -n`). A leak is acquiring a slot and never returning it; the crash (`EMFILE`/"too many open files") happens far from the leak, at the *next* allocation, often at peak load.
- **Leaks live on the error path.** The happy path closes; the exception path skips the close. The fix is structural — bind release to scope with `with` (Python), `defer` (Go), try-with-resources (Java), or RAII (C++) — so there is no error path left to forget.
- **Observe before you theorize.** `ls /proc/PID/fd | wc -l` in a loop: a *monotonic* climb is the unambiguous leak signature. `lsof -p PID` classifies it (file vs socket vs pipe), eliminating most of the codebase.
- **`CLOSE_WAIT` growth is your `close()` bug; `TIME_WAIT` growth is usually load.** Bucket sockets with `ss -tan` and read the growing state to its cause; piles of `CLOSE_WAIT` are one missing close, multiplied.
- **A drained pool looks like a deadlock but usually isn't.** Read pool metrics (`active`/`idle`/`waiting`) plus a thread dump: holders blocked *in a query* = starvation (fix the slow op); holders gone or idle = leak (fix the return path).
- **Fail fast beats block forever.** Bound every pool and queue, put a timeout on every `acquire()`, so a slow dependency produces a fast, alertable error instead of an infinite stall and a death spiral.
- **Alert on the trend, not the failure.** A gauge on FD count and a "monotonic for 30 minutes" or ">70% of limit" alert catches the leak at 2pm with runway, instead of paging you at 4:07 mid-outage.
- **Bumping the limit is a snooze button, not a cure.** Use it to survive the active incident; then find and flatline the leak.

## Further reading

- [Stop guessing: the scientific method of debugging](/blog/software-development/debugging/stop-guessing-the-scientific-method-of-debugging) — the observe → reproduce → hypothesize → bisect → fix → prevent loop this post is an instance of.
- [Hunting memory leaks and bloat](/blog/software-development/debugging/hunting-memory-leaks-and-bloat) — the sibling for when an *object*, not a descriptor, is what you forgot to release; same two-snapshot diagnostic shape.
- [Seeing what a process really does: syscall tracing](/blog/software-development/debugging/seeing-what-a-process-really-does-syscall-tracing) — `strace -f -e trace=openat,socket,close` to watch the unbalanced opens that *are* the leak.
- [Reproduce it first or you're not debugging](/blog/software-development/debugging/reproduce-it-first-or-youre-not-debugging) — repeat-until-fail loops and fault injection to make a load-dependent leak deterministic.
- [Database connection pooling](/blog/software-development/database/database-connection-pooling) — pool sizing, mechanics, and trade-offs behind the exhaustion failures in section 7.
- [Why queries are fast in dev and slow in prod](/blog/software-development/database/why-queries-are-fast-in-dev-and-slow-in-prod) — the missing-index full scans that cause pool *starvation*.
- [Debugging production at scale](/blog/software-development/system-design/debugging-production-at-scale) — how FD and pool exhaustion cascade into outages across a service fleet, and the retry-storm firebreaks.
- The Linux `lsof(8)`, `ss(8)`, and `getrlimit(2)`/`RLIMIT_NOFILE` man pages; the HikariCP docs on `leakDetectionThreshold` and `connectionTimeout`; and *Debugging* by David Agans (rule 3: "quit thinking and look" — exactly the `/proc`-first discipline here).
