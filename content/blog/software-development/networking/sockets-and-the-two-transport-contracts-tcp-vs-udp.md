---
title: "Sockets and the Two Transport Contracts: TCP vs UDP"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn exactly what the socket boundary promises, how TCP and UDP differ, how to read Linux socket state, and when rebuilding transport over UDP is justified."
tags:
  [
    "networking",
    "distributed-systems",
    "sockets",
    "tcp",
    "udp",
    "linux",
    "non-blocking-io",
    "congestion-control",
    "path-mtu",
    "quic",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/sockets-and-the-two-transport-contracts-tcp-vs-udp-1.webp"
---

A service can be healthy at the HTTP layer and still be stuck at its socket boundary. One worker waits forever in `recv`. Another process is writable according to `epoll`, yet its send queue keeps growing. A UDP migration removes connection setup, then quietly adds packet loss recovery, pacing, replay protection, and path-size bugs to the application. These failures look unrelated until we draw the boundary between code and kernel.

![The socket boundary between application intent and kernel transport state](/imgs/blogs/sockets-and-the-two-transport-contracts-tcp-vs-udp-1.webp)

The diagram above is the mental model: a socket is not the network, and it is not merely a file descriptor. It is the API boundary where an application states intent while the kernel applies a transport contract. The application supplies bytes or messages, endpoint addresses, flags, buffer capacity, and deadlines. Below that line, the kernel owns protocol state, queues, packetization, routing, and device transmission. The peer has the same boundary in reverse.

This post sits between the series [end-to-end request map](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) and the later deep dives into [the TCP handshake](/blog/software-development/networking/the-tcp-handshake-and-what-it-costs-you) and [the senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model). Here we need one narrower skill: look at an application call, identify which promise belongs to the kernel and which promise still belongs to the protocol designer, then choose the measurement that can falsify our first guess.

> A socket call tells the kernel what we want next. It does not prove what the peer received, when it received it, or what message the peer believes those bytes represent.

## The socket is the contract boundary

The Linux [`socket(7)` manual](https://man7.org/linux/man-pages/man7/socket.7.html) calls sockets the uniform interface between a user process and protocol stacks in the kernel. That phrasing is precise. `socket`, `bind`, `connect`, `listen`, `accept`, `send`, and `recv` are operations on local kernel objects. Some operations cause packets. Some only change local state. None lets application code reach through the network and manipulate the peer directly.

Consider the familiar call:

```go
conn, err := net.DialTimeout("tcp", "10.77.0.2:8080", 750*time.Millisecond)
if err != nil {
	return fmt.Errorf("dial backend: %w", err)
}
defer conn.Close()

if err := conn.SetDeadline(time.Now().Add(2 * time.Second)); err != nil {
	return fmt.Errorf("set deadline: %w", err)
}
if _, err := io.WriteString(conn, "GET /health HTTP/1.1\r\nHost: lab\r\n\r\n"); err != nil {
	return fmt.Errorf("write request: %w", err)
}
```

`net.DialTimeout` eventually asks the kernel to create and connect a stream socket. A successful TCP `connect` means the kernel completed TCP connection establishment with a peer endpoint. It does not mean an HTTP server accepted the connection in user space, parsed a request, passed a readiness check, or can reach its database. Those are higher-level contracts. The networking series owns the wire behavior; [health checks and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing) owns the application policy above it.

The same distinction applies to `write`. A successful return reports how many bytes the local kernel accepted from the caller. Those bytes may still be waiting in the local send buffer. TCP may need to retransmit them. The receiver may acknowledge them before its application reads them. If the application needs proof of durable processing, it needs an application response with the right semantics. TCP acknowledgements are not business acknowledgements.

### Five local objects hide behind one integer

The file descriptor in a Unix process is a small integer that indexes an entry in the process's descriptor table. That entry refers to an open file description, which refers to a socket object with protocol-specific state and queues. The socket is bound, explicitly or automatically, to a local address and port. A route chooses an egress path. The network device queue eventually carries packets. Treating the descriptor as "the connection" compresses all of that state into a convenient but dangerous phrase.

This compression explains several production surprises:

- Duplicated descriptors can refer to the same underlying socket state.
- A TCP listening socket and each accepted TCP connection are different sockets.
- A connected UDP socket is still datagram transport. `connect` selects a default peer and enables local filtering and error association; it does not perform a TCP-style handshake.
- Closing a descriptor does not necessarily imply that every byte has reached or been processed by the peer.
- A process can be blocked even though the network is idle, because it is waiting for local readiness or buffer space.

The boundary is useful precisely because it narrows responsibility. When a call returns `EAGAIN`, we do not start with BGP. When `ss` shows a growing receive queue, we do not first blame the remote sender. When UDP delivers a truncated message because the receive buffer passed to `recvmsg` was too small, retransmission in the network would not fix the application bug.

## One API, two different contracts

![TCP byte-stream semantics compared with UDP datagram semantics](/imgs/blogs/sockets-and-the-two-transport-contracts-tcp-vs-udp-2.webp)

TCP and UDP both fit through the socket API, but their promises are not small variations on the same service. TCP provides a reliable, ordered, full-duplex byte stream between two endpoints. UDP provides individual datagrams, each with source and destination ports plus a checksum, without built-in delivery, ordering, duplicate suppression, or congestion control.

The current TCP specification is [RFC 9293, published in August 2022](https://www.rfc-editor.org/rfc/rfc9293.html). Linux's [`tcp(7)` documentation](https://man7.org/linux/man-pages/man7/tcp.7.html) states the application-visible point plainly: TCP retransmits lost packets and delivers data in order, but it does not preserve record boundaries. If a client makes three writes of 5, 7, and 4 bytes, the server does not acquire three records. It acquires a stream of 16 ordered bytes. Reads might return 16 bytes once, 8 bytes twice, or another partition allowed by the receive buffer and timing.

That gives us the first rule of stream programming:

> One `write` is not one `read`. Message framing belongs to the application protocol.

A length prefix, delimiter, fixed-size record, or self-describing format tells the receiver where one application message ends. Without framing, the receiver cannot distinguish `"12" + "345"` from `"123" + "45"`. TCP faithfully transports the byte sequence and faithfully refuses to invent the missing semantics.

UDP makes the opposite boundary visible. Linux's [`udp(7)` documentation](https://man7.org/linux/man-pages/man7/udp.7.html) says each receive operation returns one packet. If the caller's buffer is too small, the packet is truncated and `MSG_TRUNC` is set. The unread tail is not left for the next call. This is message preservation with a sharp edge: the receive operation must provide enough capacity or explicitly detect truncation.

### `SOCK_STREAM` means continuity, not messages

Suppose an application sends a header containing a 32-bit network-order length followed by that many payload bytes. The receiver must loop for both pieces because either read can return short:

```go
func readFrame(r io.Reader, max uint32) ([]byte, error) {
	var length uint32
	if err := binary.Read(r, binary.BigEndian, &length); err != nil {
		return nil, fmt.Errorf("read frame length: %w", err)
	}
	if length > max {
		return nil, fmt.Errorf("frame length %d exceeds limit %d", length, max)
	}

	payload := make([]byte, length)
	if _, err := io.ReadFull(r, payload); err != nil {
		return nil, fmt.Errorf("read %d-byte frame: %w", length, err)
	}
	return payload, nil
}
```

`io.ReadFull` is doing important protocol work here. A single `Read` is allowed to return fewer bytes than requested with no network failure. The maximum frame size is equally important. A peer-controlled length without a bound turns framing into a memory exhaustion primitive.

TCP's ordering guarantee also has a cost. If a byte range is missing, later bytes cannot be delivered as the same ordered stream until the gap is repaired. That is correct for a database protocol whose later message depends on earlier state. It can be counterproductive for independent, expiring updates where fresh state is more valuable than a late stale update. The transport contract must match the application's dependency structure.

### EOF is part of the stream contract

TCP represents an orderly end of the peer's sending direction as end-of-file to the receiving application. On Unix-like systems, a stream `recv` returning zero after earlier data means the peer has performed an orderly shutdown of that direction. It does not mean "try again later." `EAGAIN` means no progress is currently available on a non-blocking descriptor. Confusing the two creates either busy loops or connections that never release.

TCP is full duplex, so each direction can close independently. An application can call `shutdown(SHUT_WR)` to send no more bytes while continuing to receive. Protocols that use EOF as message framing rely on this half-close. Connection pools generally cannot, because an EOF-framed response consumes the connection. Once again, an API event acquires meaning only through the application protocol.

An abrupt reset is different from orderly EOF. A reset tells the local stack that the connection is no longer valid, but it still does not identify which remote application operation committed before the failure. If a client sent a payment request and then saw a reset, transport cannot decide whether retry is safe. That is why idempotency and ambiguous-outcome handling sit above TCP.

UDP has no stream EOF. A zero-length UDP datagram is a valid message, not proof that a peer closed. Silence is also ambiguous: the peer may have stopped, the request may be lost, the response may be lost, or a policy device may be dropping traffic. A UDP protocol that needs liveness defines its own challenge, response, lease, or idle timeout. The timeout proves only that the chosen evidence did not arrive within the chosen interval.

### `SOCK_DGRAM` means messages, not reliability

With UDP, one successful `sendto` submits one datagram to the local stack. One successful `recvfrom` returns at most one datagram and identifies its source. Those statements are local API semantics. They do not imply that every submitted datagram reaches the peer. The network may drop, duplicate, or reorder packets. A receiver that requires exactly-once effects still needs identifiers, replay or duplicate handling, and an application acknowledgement tied to the effect.

A connected UDP socket often confuses engineers because the call is named `connect`. [RFC 8085, published in March 2017](https://www.rfc-editor.org/rfc/rfc8085.html) explains that UDP `connect` is a local operation that binds the socket to a peer address pair for sending, filtering, and asynchronous error delivery. No setup packet is required. A connected UDP socket therefore has a peer, but it does not have TCP connection state or TCP delivery guarantees.

### The contracts side by side

| Question | TCP stream socket | UDP datagram socket |
| --- | --- | --- |
| What does one receive return? | Some currently available ordered bytes | At most one datagram |
| Are application write boundaries preserved? | No | Yes, for a delivered and non-truncated datagram |
| Is loss recovered by the transport? | Yes, while the connection remains viable | No |
| Can data arrive to the application out of order? | No | Yes |
| Can duplicates reach the application? | Not as duplicate stream bytes | Yes |
| Is congestion control built in? | Yes | No |
| Does `connect` imply a handshake? | Yes | No |
| Who defines message framing? | Application protocol | Datagram boundary, plus application structure |

The table contains qualitative protocol properties, not benchmark results. There is no context-free statement that UDP is faster. Removing work can reduce delay, but rebuilding equivalent guarantees adds work back. A UDP protocol can intentionally omit stale-data recovery and win for that workload. A careless UDP loop can also overload a path, fragment packets, and become less reliable than the TCP design it replaced.

## Blocking and non-blocking describe waiting, not transport

<figure class="blog-anim">
<svg viewBox="0 0 900 430" role="img" aria-label="A blocking worker waits on delayed socket A while a non-blocking readiness loop serves ready sockets B and C before returning to A" style="width:100%;height:auto;max-width:900px">
<style>
.sock5-title{font:700 20px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.sock5-label{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.sock5-note{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.sock5-card{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.sock5-wait{fill:none;stroke:var(--border,#d1d5db);stroke-width:8;stroke-linecap:round}.sock5-run{fill:none;stroke:var(--accent,#6366f1);stroke-width:8;stroke-linecap:round;stroke-dasharray:38 330}.sock5-dot{fill:var(--accent,#6366f1)}
@keyframes sock5-block{0%,72%{stroke-dashoffset:0;opacity:.25}78%,100%{stroke-dashoffset:-330;opacity:1}}@keyframes sock5-loop{0%{stroke-dashoffset:0}28%{stroke-dashoffset:-105}55%{stroke-dashoffset:-210}78%,100%{stroke-dashoffset:-330}}@keyframes sock5-ready{0%,68%{opacity:.2}76%,100%{opacity:1}}
.sock5-blocking{animation:sock5-block 10s ease-in-out infinite}.sock5-event{animation:sock5-loop 10s ease-in-out infinite}.sock5-a-ready{animation:sock5-ready 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.sock5-blocking,.sock5-event,.sock5-a-ready{animation:none}.sock5-blocking,.sock5-event{stroke-dasharray:none}.sock5-a-ready{opacity:1}}
</style>
<text class="sock5-title" x="40" y="34">Same delayed bytes, different scheduling contract</text>
<text class="sock5-label" x="115" y="92">blocking worker</text><rect class="sock5-card" x="40" y="112" width="150" height="84" rx="10"/><text class="sock5-label" x="115" y="146">socket A read</text><text class="sock5-note" x="115" y="171">worker parked</text>
<path class="sock5-wait" d="M215 154 H810"/><path class="sock5-run sock5-blocking" d="M215 154 H810"/><circle class="sock5-dot sock5-a-ready" cx="810" cy="154" r="10"/><text class="sock5-note" x="512" y="132">A is delayed; this execution context cannot serve B or C</text>
<text class="sock5-label" x="115" y="274">readiness loop</text><rect class="sock5-card" x="40" y="294" width="150" height="84" rx="10"/><text class="sock5-label" x="115" y="328">poll readiness</text><text class="sock5-note" x="115" y="353">advance ready work</text>
<path class="sock5-wait" d="M215 336 H810"/><path class="sock5-run sock5-event" d="M215 336 H810"/>
<rect class="sock5-card" x="260" y="286" width="130" height="98" rx="10"/><text class="sock5-label" x="325" y="326">socket B</text><text class="sock5-note" x="325" y="350">ready now</text>
<rect class="sock5-card" x="460" y="286" width="130" height="98" rx="10"/><text class="sock5-label" x="525" y="326">socket C</text><text class="sock5-note" x="525" y="350">ready now</text>
<rect class="sock5-card" x="660" y="286" width="150" height="98" rx="10"/><text class="sock5-label" x="735" y="326">socket A</text><text class="sock5-note" x="735" y="350">ready later</text>
</svg>
<figcaption>The same delayed bytes park one blocking worker, while a readiness loop serves sockets B and C before returning to socket A.</figcaption>
</figure>

Blocking mode answers a local scheduling question: if an operation cannot make progress now, may this calling execution context sleep until it can? Non-blocking mode answers the same question differently: return immediately with progress or with a status such as `EAGAIN`, then let the program decide what else to do. Neither mode changes TCP into UDP or changes delivery guarantees.

The Linux [`socket(7)` manual](https://man7.org/linux/man-pages/man7/socket.7.html) documents the mechanism. Setting `O_NONBLOCK` on the file descriptor makes operations that would wait usually return `EAGAIN`; a non-blocking `connect` reports `EINPROGRESS`. The program can then wait for readiness using `poll`, `select`, or `epoll`.

### Blocking is often the clearest correct design

A blocking socket is not inherently slow. A thread that owns one connection, has explicit deadlines, and performs simple request-response work can be easy to reason about. Modern runtimes may schedule many language-level tasks over fewer OS threads while presenting a blocking-looking API. The operational question is not whether the source contains `await` or `epoll`. It is whether waiting work consumes a scarce execution resource and whether every wait has a bounded lifetime.

The failure mode appears when an unbounded call pins a bounded worker pool. If 200 workers can each block forever in `recv`, then 200 silent peers can consume the whole pool. The network can be perfectly healthy while new work queues behind unavailable workers. Deadlines, cancellation, admission control, and bounded concurrency are the controls. [Rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) owns the system-wide policy; the socket layer exposes the queue and readiness facts that policy acts on.

### Non-blocking is a state machine, not a performance flag

Non-blocking I/O trades sleeping stacks for explicit state. The program must retain partial input, partial output, framing state, timeouts, and ownership across callbacks or tasks. A write-ready notification does not promise that an arbitrarily large buffer fits. A read-ready notification does not promise that a complete application frame is available. It promises that an attempted operation can make some progress or expose a condition such as EOF or an error.

Edge-triggered `epoll` adds another contract. The [`epoll(7)` manual](https://man7.org/linux/man-pages/man7/epoll.7.html) recommends non-blocking descriptors and draining reads or writes until `EAGAIN` before waiting again. If code reads only one chunk after an edge notification and leaves more data buffered, it may wait indefinitely for an edge that already happened. Level-triggered operation is more forgiving because readiness continues to be reported while the condition remains true, though poor event-loop discipline can still cause unfairness.

```c
for (;;) {
    int n = epoll_wait(epfd, events, MAX_EVENTS, timeout_ms);
    if (n < 0 && errno == EINTR) continue;
    if (n < 0) die("epoll_wait");

    for (int i = 0; i < n; i++) {
        int fd = events[i].data.fd;
        for (;;) {
            ssize_t got = recv(fd, buf, sizeof(buf), 0);
            if (got > 0) {
                consume_bytes(fd, buf, (size_t)got);
                continue;
            }
            if (got == 0) {
                close_connection(fd);
                break;
            }
            if (errno == EAGAIN || errno == EWOULDBLOCK) break;
            if (errno == EINTR) continue;
            fail_connection(fd, errno);
            break;
        }
    }
}
```

That loop is incomplete as a server, but its receive rule is real. Production code also limits work per ready descriptor so one busy connection cannot starve the rest. It treats EOF separately from `EAGAIN`, maintains frame parsing across chunks, and unregisters descriptors before releasing connection state.

### Backpressure begins at the send call

A non-blocking send that returns a short count or `EAGAIN` is not an invitation to discard the remainder. It is the kernel saying that the local send path currently lacks space. The application must retain unsent bytes, subscribe to write readiness, and cap the retained backlog. Otherwise the kernel queue merely moves into an unbounded user-space queue.

This gives us a causal chain worth memorizing:

1. The peer or path drains bytes more slowly than the application produces them.
2. The local TCP send queue grows.
3. The socket stops accepting the full write immediately.
4. A blocking writer sleeps; a non-blocking writer sees partial progress or `EAGAIN`.
5. If the application ignores this signal by buffering forever, memory becomes the next queue.

Backpressure is therefore not a library feature added above transport. It begins with finite buffers and a caller willing to respect them.

### Buffers buy time, not capacity

Socket buffers absorb short mismatches between production and consumption. They cannot repair a sustained rate mismatch. A simple explanatory model makes the cost visible:

$$
T_{\text{drain}} \approx \frac{Q \times 8}{R}
$$

Here $Q$ is queued application data in bytes, $R$ is the eventual drain rate in bits per second, and $T_{\text{drain}}$ is the queue's minimum drain time if no new data arrives. This is an approximation, not an equation specified by TCP. It ignores headers, retransmissions, congestion-window evolution, and competing traffic.

Suppose an application allows 8 MiB of pending output for one slow connection and the path drains at 10 Mbit/s. Using binary mebibytes for the queue:

$$
T_{\text{drain}} \approx \frac{8 \times 1{,}048{,}576\ \text{bytes} \times 8\ \text{bits/byte}}{10{,}000{,}000\ \text{bits/s}} = 6.71\ \text{s}
$$

That queue can make a freshly generated message wait more than 6 seconds behind old data before accounting for network round trips or retransmission. Shrinking the per-connection cap to 64 KiB yields:

$$
T_{\text{drain}} \approx \frac{65{,}536 \times 8}{10{,}000{,}000} = 0.0524\ \text{s}
$$

The smaller queue can expose overload sooner and keep stale work from dominating memory, but it also tolerates less burst. The correct cap comes from the application's freshness budget, expected drain variance, connection count, and retry behavior.

| Pending bytes | Assumed drain rate | Approximate no-new-work drain time | Source |
| --- | --- | --- | --- |
| 64 KiB | 10 Mbit/s | 52.4 ms | Derived here from $T_{\text{drain}} \approx Q \times 8 / R$ |
| 1 MiB | 10 Mbit/s | 839 ms | Derived here from the same model |
| 8 MiB | 10 Mbit/s | 6.71 s | Derived here from the same model |

At 10,000 simultaneously slow connections, an 8 MiB application backlog per connection would permit about 78.1 GiB of queued payload: ${10{,}000 \times 8\ \text{MiB} / 1{,}024 = 78.125\ \text{GiB}}$. That is a configured upper bound, not a measured resident-set size. Allocator overhead, metadata, kernel buffers, and shared representations change actual memory use. The calculation is still useful because it forces a per-connection convenience to face a fleet-wide budget.

The senior rule is to bound queues at every layer and define what happens at the bound. Options include stop reading upstream, reject new work, drop superseded updates, shed the connection, or spill to durable storage. "Keep buffering" is a decision too, usually an accidental one.

### Readiness is level information, not a work budget

One ready descriptor can contain one byte or many megabytes. An event loop that drains it without a per-turn budget can starve other ready descriptors. An event loop that reads one small chunk in edge-triggered mode can strand data without another notification. Correct implementations combine the kernel's readiness contract with an application scheduler: drain until `EAGAIN`, but rotate fairly using a ready list or explicit byte and message budgets.

The same rule applies to writes. Registering every socket permanently for write readiness often creates a hot loop because sockets are commonly writable. Subscribe when pending output exists, attempt bounded progress, retain any remainder, and remove write interest when the user-space queue becomes empty. This reduces wakeups and makes `Send-Q` plus application backlog meaningful together.

## What UDP makes the application rebuild

![Responsibilities retained by TCP and moved into a UDP-based application protocol](/imgs/blogs/sockets-and-the-two-transport-contracts-tcp-vs-udp-3.webp)

UDP is deliberately small. [RFC 768, published in August 1980](https://www.rfc-editor.org/rfc/rfc768.html) defines source port, destination port, length, checksum, and payload. That minimal contract is valuable, but it creates an engineering invoice. If the application needs a property that UDP does not provide, some layer above UDP must implement it or explicitly decide that it is unnecessary.

### Ordering requires identity and a policy

Packets can take different paths, wait in different queues, and arrive out of order. A protocol that cares about order therefore needs a sequence space. Sequence numbers alone are not enough. The receiver also needs a window defining which numbers are plausible, a duplicate policy, wraparound-safe comparisons, and a rule for gaps.

For state updates, the right rule may be "apply the newest version and discard older versions." For a file transfer, the rule may be "buffer later chunks until the missing chunk arrives." For voice, the rule may be "wait only until the playout deadline, then conceal the loss." These are different contracts. TCP chooses ordered delivery for the single byte stream. UDP lets the application choose, which is useful only when the application actually makes the choice.

Imagine position updates numbered 410, 411, and 412. If 411 is late but 412 is already usable, holding 412 for 411 increases staleness. A freshness-oriented protocol can apply 412 and discard 411 when it arrives. TCP cannot expose byte 412 ahead of a missing earlier byte within the same stream because doing so would violate its contract. The application's dependency graph, not protocol fashion, determines which behavior is correct.

### Reliability requires feedback and bounded memory

Reliable delivery needs at least four pieces:

1. A unit that can be identified, usually with packet and message identifiers.
2. Feedback indicating received or missing units.
3. A loss detector based on acknowledgements, time, or both.
4. A retransmission policy with retry and lifetime limits.

It also needs careful state cleanup. A sender that retains every unacknowledged message forever has converted packet loss into unbounded memory growth. A receiver that remembers every identifier forever has the same problem in reverse. Timeouts must be tied to round-trip behavior rather than chosen as a magic constant, and retries must stop when the application value expires.

Reliability is not synonymous with retransmitting the same bytes. Suppose an application sends a query, times out, and sends it again with a new identifier. The server may execute both. Transport-level duplicate suppression cannot infer that the two requests have the same business effect. Idempotency keys and application result semantics still belong above the transport.

### Congestion control is a safety property

An application can send UDP as fast as the local host accepts datagrams. That does not mean the path can carry them. When aggregate offered load exceeds a bottleneck's capacity, queues grow and packets drop. Senders that do not reduce load can drive the path toward congestion collapse and can take unfair capacity from responsive flows.

[RFC 8085](https://www.rfc-editor.org/rfc/rfc8085.html) makes congestion control a central requirement for UDP applications on the Internet. The RFC identifies two goals: prevent congestion collapse and establish reasonable fairness among competing traffic. A custom reliable UDP protocol therefore needs pacing, a congestion window or rate model, round-trip estimation, loss or explicit-congestion feedback, and recovery behavior. Implementing acknowledgements without congestion response is not a finished transport.

There is one legitimate shortcut: a tightly controlled environment with a known fixed rate and bounded traffic volume may use a simpler policy. Even there, the design must state the bound and enforcement mechanism. "It is inside our network" is not congestion control. Internal links also have finite queues, failures, and competing tenants.

### Path MTU discovery becomes protocol work

Maximum transmission unit, or MTU, is the largest IP packet a link can carry without fragmentation. The path MTU is the smallest MTU across all links on the route. A UDP application that emits datagrams larger than the path can carry must deal with fragmentation or loss.

For a simple derived example, assume an Ethernet-like path MTU of 1,500 bytes, an IPv4 header with no options of 20 bytes, and the fixed UDP header of 8 bytes. The largest UDP payload that fits without IP fragmentation is approximately:

$$
1{,}500\ \text{bytes} - 20\ \text{bytes} - 8\ \text{bytes} = 1{,}472\ \text{bytes}
$$

For IPv6 without extension headers, the base IPv6 header is 40 bytes, so the corresponding payload is:

$$
1{,}500\ \text{bytes} - 40\ \text{bytes} - 8\ \text{bytes} = 1{,}452\ \text{bytes}
$$

These are derived examples, not universal safe payloads. Tunnels and extension headers reduce the effective size, and a different link on the path can have a smaller MTU. [RFC 8085](https://www.rfc-editor.org/rfc/rfc8085.html) recommends avoiding UDP datagrams that exceed the path MTU because losing one IP fragment loses the whole original datagram, while some NATs and firewalls drop fragments. [RFC 8899, published in September 2020](https://www.rfc-editor.org/rfc/rfc8899.html) specifies Datagram Packetization Layer Path MTU Discovery so datagram protocols can probe usable sizes without trusting only ICMP feedback.

TCP normally hides packetization from the application. It negotiates a maximum segment size and adapts segment payload to path information. A UDP application owns its message size. If an application message is larger than its safe datagram size, it needs application fragmentation, identifiers, reassembly limits, independent recovery, and defenses against incomplete-fragment memory exhaustion.

Consider a protocol that caps its application fragment payload at 1,200 bytes and needs to carry a 4,000-byte message. Ignoring the protocol's own per-fragment header for the moment, the application needs:

$$
N = \left\lceil \frac{4{,}000}{1{,}200} \right\rceil = 4\ \text{datagrams}
$$

The first three carry 1,200 payload bytes each and the fourth carries 400. The receiver needs a message identifier, fragment positions, total-length validation, a reassembly timeout, and a memory bound. If the protocol adds a 24-byte fragment header, that header must fit inside the chosen payload budget rather than being added beyond the safe datagram size.

Fragmentation also multiplies exposure to loss. As an explanatory probability model, assume each of the four datagrams has an independent 1% loss probability. This independence assumption is often unrealistic because real losses can be bursty, so the calculation is intuition rather than a path prediction. The probability that all four arrive is:

$$
P(\text{complete}) = (1 - 0.01)^4 = 0.9606
$$

The probability that at least one is missing is therefore about ${1 - 0.9606 = 0.0394}$, or 3.94%. If the whole message must be retransmitted after any missing fragment, one loss repeats already delivered work. A better design can acknowledge fragments or ranges and retransmit only missing pieces, but that requires more identifiers, timers, and state. This is how a "simple UDP message" grows into a transport.

| Application message | Fragment payload cap | Datagram count | Complete probability under the illustrative model | Source |
| --- | --- | --- | --- | --- |
| 1,000 bytes | 1,200 bytes | 1 | 99.00% | Derived here with independent 1% per-datagram loss |
| 4,000 bytes | 1,200 bytes | 4 | 96.06% | Derived here from $(1 - 0.01)^4$ |
| 12,000 bytes | 1,200 bytes | 10 | 90.44% | Derived here from $(1 - 0.01)^{10}$ |

Do not use the table as a benchmark. Its job is to expose the shape of the risk: more required datagrams create more opportunities for an incomplete message, and correlated loss can be worse than the independent model. The production answer is to measure the actual path, keep messages independently useful where possible, use bounded selective recovery when completeness matters, and adapt message size through PMTU discovery.

The reassembly receiver must also reject impossible declarations early. A fragment claiming a 2 GiB total message should not reserve 2 GiB. Limit concurrent incomplete messages per peer and globally, expire them, authenticate enough metadata to prevent cheap spoofed allocations, and count evictions. These are application-layer controls made necessary by choosing a datagram contract.

### Security and connection identity do not appear for free

UDP has no connection handshake that proves return reachability before a server replies. A small request with a spoofed source can induce a larger response toward a victim unless the protocol validates the address or limits amplification. A production design may need cookies or tokens, anti-replay state, authentication, encryption, key rotation, and connection identifiers that survive address changes.

The word "connectionless" should not mislead us. A useful protocol over UDP often creates logical connection state in user space. It simply does not inherit TCP's kernel state machine. QUIC is the clearest example: its connection IDs, packet numbers, acknowledgement ranges, cryptographic handshake, flow control, congestion control, and path validation form a substantial transport above UDP.

## When each trade is correct

![Decision matrix for TCP streams and UDP-based protocols](/imgs/blogs/sockets-and-the-two-transport-contracts-tcp-vs-udp-4.webp)

The first decision is not "which protocol is faster?" It is "what must be true when the receiver acts?" Once we state that contract, transport selection becomes much less mystical.

Choose a TCP stream when all bytes matter, order matters, and established congestion behavior is more valuable than custom per-message delivery. Database connections, most HTTP/1.1 and HTTP/2 traffic, SSH, and replication logs fit naturally. You still need application framing, deadlines, authentication, and overload policy. TCP solves transport reliability, not application correctness.

Choose plain UDP when each message is independently useful, loss is acceptable or handled by sampling, and bounded timeliness matters more than complete history. Metrics samples, discovery probes, or live media units can fit, depending on their security and congestion requirements. Even low-volume UDP needs explicit payload bounds and source validation.

Choose a sophisticated UDP-based transport when you need properties that kernel TCP cannot expose cleanly and you are prepared to own a transport implementation. Independent streams, user-space evolution, connection migration, or partial reliability can justify the cost. In most product teams, using an established implementation such as QUIC is safer than inventing one.

| Requirement | Prefer | Reason | New failure to plan for |
| --- | --- | --- | --- |
| Ordered, complete byte sequence | TCP | Kernel provides retransmission, ordering, flow control, and congestion control | Head-of-line delay and missing application framing |
| Independent expiring updates | UDP with an application policy | New state need not wait for stale state | Reordering, duplicates, pacing, and authentication |
| Many independent reliable streams | QUIC or another established multiplexed transport | Stream loss can be isolated above UDP packet delivery | More complex user-space transport and UDP path blocking |
| One-shot request across arbitrary networks | Usually TCP/TLS or QUIC | Mature fallback and security ecosystem | Setup cost, retry ambiguity, and middleboxes |
| Tiny bounded probes in a controlled domain | UDP | Minimal exchange and preserved message boundary | Amplification, false timeout conclusions, and loss |

This is a transport decision, not an architecture decision. Service boundaries, retry budgets, and failure isolation belong in the broader [timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) design. The socket choice determines what evidence and failure modes that design receives.

### A practical review checklist

Before approving a UDP protocol, ask for written answers to these questions:

- What identifies a message, packet, connection, and peer?
- Which data may be dropped, reordered, duplicated, or superseded?
- How does the sender learn delivery, loss, or congestion?
- What bounds retransmission state, receive reassembly, and retry lifetime?
- How is offered load paced and reduced under congestion?
- How is the maximum safe datagram size discovered and updated?
- How does the server validate a source before amplifying a response?
- What happens when UDP is blocked but TCP is available?
- Which counters distinguish network loss from application discard?

If the answer to several questions is "we will add that later," the design is not choosing UDP's trade. It is postponing the transport design.

The same review should name observability before rollout. Count submitted, received, truncated, rejected, duplicate, late, retransmitted, and expired datagrams separately. Record congestion and pacing state for protocols that implement them. For TCP, separate connection establishment, stream read or write timeout, reset, orderly EOF, and application protocol error. One generic `network_error` counter erases the contract boundary and makes the next incident slower.

Fallback also belongs in the contract. If a client can use both a UDP-based transport and TCP, specify when it tries each path, how long the first attempt may delay the second, and how success is cached. A fallback that waits through several opaque UDP timeouts can be correct eventually and still violate the user's latency budget. Conversely, racing transports without resource bounds can double load during an outage. The transport decision includes the transition policy, not only the happy-path socket type.

## A public case: QUIC rebuilt transport over UDP

![QUIC transport machinery layered over UDP sockets](/imgs/blogs/sockets-and-the-two-transport-contracts-tcp-vs-udp-5.webp)

Google's [SIGCOMM 2017 paper, presented August 21–25, 2017](https://research.google/pubs/the-quic-transport-protocol-design-and-internet-scale-deployment/), is a useful public case because it does not treat UDP as a magic performance bit. The authors describe QUIC as an encrypted, multiplexed transport designed for HTTPS and for rapid deployment and evolution in user space. At the time described by the paper, Google had deployed it across thousands of servers and estimated that it carried 7% of Internet traffic. Those numbers belong to that 2017 deployment and should not be generalized to today's Internet.

The trigger was not that TCP had stopped working. Changing transport behavior in kernel TCP and deploying it across client and server operating systems was slow. UDP offered a widely available socket substrate on which both endpoints could evolve a transport in user space. That deployment advantage moved work upward rather than eliminating it.

The resulting protocol restored the machinery a production transport needs: connection establishment, cryptographic protection, packet numbers, acknowledgements, loss detection, retransmission of information, flow control, congestion control, and multiplexed streams. The standardized result, [RFC 9000 published in May 2021](https://www.rfc-editor.org/rfc/rfc9000.html), describes QUIC as a UDP-based multiplexed and secure transport. Its companion recovery specification defines loss detection and congestion control. RFC 9000 also requires path-size behavior, including a smallest allowed maximum UDP payload of 1,200 bytes for QUIC and guidance for PMTU discovery. That 1,200-byte value is a QUIC protocol requirement, not a universal recommendation that every UDP application should copy.

The causal chain is the lesson:

1. UDP made user-space deployment through existing socket APIs practical.
2. UDP did not provide reliable streams, congestion response, security, or path validation.
3. QUIC implemented those properties as an integrated transport.
4. Moving the implementation boundary enabled faster protocol evolution and independent streams, but also created substantial user-space complexity.

The blast-radius multiplier for a naive imitation is believing step 1 provides the value without steps 2 and 3. A home-grown protocol can look excellent on a quiet LAN and fail on lossy, reordered, tunneled, or adversarial paths. QUIC's history argues for UDP as a substrate when the new transport has a specific reason to exist. It does not argue for replacing TCP with raw datagrams around arbitrary RPCs.

The transferable control is a responsibility ledger. For every TCP property removed, name the replacement mechanism, the observable counter, the state bound, and the failure behavior. If the protocol intentionally omits a property, state which application invariant makes the omission safe.

## Read `ss` output line by line

![A decision tree for reading Linux ss socket state](/imgs/blogs/sockets-and-the-two-transport-contracts-tcp-vs-udp-6.webp)

Linux `ss` dumps socket statistics from the kernel. Its power is not the volume of fields. Its power is that it lets us inspect the exact boundary we have been discussing. The [`ss(8)` manual](https://man7.org/linux/man-pages/man8/ss.8.html) documents protocol filters such as `-t` for TCP and `-u` for UDP, `-l` for listening sockets, `-p` for process ownership, `-n` for numeric endpoints, and `-i` for internal TCP information.

Start with a narrow command:

```bash
sudo ss -Hntpi state established '( sport = :8080 or dport = :8080 )'
sudo ss -Hlnpt 'sport = :8080'
sudo ss -Huanp 'sport = :8080'
```

`-H` removes the header, which is convenient for scripts but means we must know the columns. During an incident, run once without `-H` before copying fields into an automation. `-n` prevents name resolution from adding latency or misleading service names. `-p` may require privilege to reveal another process.

### 1. Read `Netid` and `State`

`Netid` tells us which socket table produced the row, commonly `tcp` or `udp` here. `State` must be interpreted through that protocol. TCP `LISTEN`, `ESTAB`, `SYN-SENT`, `CLOSE-WAIT`, and `TIME-WAIT` name state-machine positions. A large `CLOSE-WAIT` population means the peer closed and the local application has not closed its side. It is not proof of packet loss.

UDP has no TCP connection state machine. A connected UDP socket can appear with an established-looking association in tools because a default peer is recorded locally. That is still not evidence of a handshake or delivery.

### 2. Read `Recv-Q` in context

For an established socket, `Recv-Q` represents data queued for the local application to read. A growing value means bytes or datagrams have reached the local kernel faster than the process consumes them. The first investigation is local scheduling, application work, receive-buffer pressure, and event-loop correctness.

For a listening TCP socket, queue columns have different semantics related to pending connections and the configured backlog. Do not apply established-socket interpretation to a listener. The later handshake post examines those queues and their counters in detail.

For UDP, a receive queue can contain multiple datagrams. Application buffer sizing still matters: the kernel can queue a large datagram, then `recvmsg` with a small user buffer can truncate it. `Recv-Q` proving arrival does not prove correct consumption.

### 3. Read `Send-Q` in context

For an established TCP socket, a persistent send queue means locally written bytes have not yet been acknowledged and retired. Possible causes include a slow reader, a constrained path, receiver flow control, congestion, or loss. One snapshot cannot distinguish them. `ss -ti` exposes TCP internals that refine the hypothesis.

A transient nonzero value during active transfer is normal. The diagnostic signal is persistence or growth correlated with application latency. Compare repeated snapshots and pair them with the peer's receive behavior. If user-space buffering grows while `Send-Q` remains small, the backpressure problem is above the socket. If `Send-Q` grows, the kernel boundary is participating.

### 4. Read local and peer endpoints

`Local Address:Port` answers where this socket is bound. `Peer Address:Port` answers which remote endpoint is associated. Wildcards such as `0.0.0.0:8080` or `[::]:8080` on a listener mean the bind covers multiple local addresses in that family. They do not mean packets are sent to address zero.

Endpoint identity also prevents a common incident error: inspecting the right port in the wrong network namespace. Run `ip netns exec s ss ...` for the lab server or enter the target container's network namespace in production. A host-level empty result does not prove the container has no socket.

### 5. Read process ownership

With sufficient permission, `users:(...)` associates a socket with process names, PIDs, and descriptors. This is the bridge back to code. Confirm that the expected binary owns the listener and that there are not stale processes sharing a port through `SO_REUSEPORT`.

Ownership can be absent because of permissions, timing, kernel interfaces, or sockets that outlive an observed process transition. Treat absence as missing evidence, not proof that no process exists.

### 6. Read TCP internals only after the row makes sense

`ss -ti` can show fields such as round-trip estimates, retransmission state, congestion-control name, congestion window, receiver window, pacing rates, and byte counters. Field availability varies with kernel and iproute2 version. Record `uname -r` and `ss -V` when comparing systems.

Read these fields as a coherent state, not as isolated alarms. A round-trip estimate needs units and sampling context. A congestion window only matters with segment size, receiver window, bytes in flight, and the application's demand. Retransmission evidence supports loss or reordering, but an empty receive queue with high application latency can still point above transport.

| Observation | First hypothesis | Discriminating next check | Source |
| --- | --- | --- | --- |
| Established TCP `Recv-Q` grows across snapshots | Local application is not draining bytes | Inspect thread or event-loop state and capture `recv` activity | Reproducible with the lab below |
| Established TCP `Send-Q` grows across snapshots | Peer or path is not retiring bytes | Read `ss -ti`, peer `Recv-Q`, and targeted packet acknowledgements | Reproducible with a controlled slow reader |
| Many `CLOSE-WAIT` sockets | Local application has not closed after peer EOF | Map owners with `ss -p`, then inspect close paths | TCP state semantics in RFC 9293 |
| UDP row has a peer endpoint | Local socket has a default peer | Verify socket type and remember there was no handshake | RFC 8085, March 2017 |
| UDP payload is truncated | User receive buffer is smaller than one datagram | Use `recvmsg`, inspect `MSG_TRUNC`, and increase bounded capacity | Linux `udp(7)` and POSIX `recvmsg` |

Every numeric or state entry in this table is qualitative and sourced. We have not pasted fabricated command output. The lab will generate output on the reader's own kernel, where queue values and internal fields are real.

## Diagnose from the boundary outward

The most efficient socket investigations move from the application call to local kernel state, then to packets, then to the peer. Starting with a broad dashboard often mixes several layers into one symptom. A tight sequence gives each hypothesis a chance to fail quickly.

### If `connect` is slow or fails

First identify the socket type and the exact endpoint. A TCP `connect` can wait on route resolution, neighbor discovery, SYN delivery, the peer's response, or local resource limits. A UDP `connect` usually records local association state and can succeed without proving that a peer exists. Treating those two results as equivalent reachability tests is a category error.

For TCP, inspect the state with a focused `ss` filter while the call is pending. `SYN-SENT` tells us the local kernel has begun active opening. A packet capture can then answer whether SYN leaves, whether SYN-ACK returns, and whether a reset arrives. If there is no socket row, inspect the application call and local error before chasing the path.

For UDP, use an application request and response, or a protocol-defined validation exchange. A successful local send reports submission, not peer reachability. ICMP errors may be delivered asynchronously, filtered, delayed, or associated differently depending on whether the socket is connected. The absence of an error is not an acknowledgement.

### If reads stall

Ask four questions in order:

1. Does the application have a deadline or cancellation path?
2. Does `ss` show bytes or datagrams in `Recv-Q`?
3. Is the process scheduled and calling `recv` on the expected descriptor?
4. Does a targeted capture show payload arriving at the host?

If `Recv-Q` grows, the path delivered data to the local socket and the process is not draining it quickly enough. If `Recv-Q` stays empty while capture shows packets, check tuple matching, namespace, firewall, checksum, and protocol state. If capture shows nothing, move outward to route and peer. Each observation narrows the layer.

A stream parser can also appear stalled because it waits for framing bytes that the sender never emits. This is not TCP losing a message boundary. TCP never promised one. Log the parser's expected length and accumulated byte count, with payload logging disabled or carefully redacted. Packet captures can contain credentials, tokens, and personal data, so capture only the tuple and length required for the question.

### If writes stall or memory grows

Compare three queues: the application's pending output, the socket `Send-Q`, and the peer's receive queue. A growing application queue with a small kernel queue can indicate that the event loop is not attempting writes or that per-connection scheduling is unfair. A growing kernel send queue points toward peer consumption, receiver flow control, loss, congestion, or path capacity. A growing peer receive queue points back to the peer application.

This queue chain prevents a common operational mistake. Increasing `SO_SNDBUF` may temporarily hide pressure by allowing more bytes into the kernel, but it does not make the peer or path faster. It can increase memory use and the amount of stale work waiting behind a slow connection. Buffer tuning is a capacity decision with latency and memory consequences, not a generic fix.

### If UDP works in the lab but fails across the Internet

Vary one path property at a time. Test payload size before blaming random loss. Test return reachability before adding retries. Check whether the network blocks or rate-limits UDP. Record the effective source and destination tuple after NAT. Then test reordering, duplication, and loss with a controlled impairment.

The first payload-size calculation is only a bound. A 1,472-byte UDP payload fits a 1,500-byte IPv4 path with minimal headers, but a tunnel can lower the path MTU. If 1,200-byte payloads work and larger payloads disappear, investigate PMTU behavior and ICMP handling. Do not respond by adding unlimited retries because repeated oversized datagrams reproduce the same failure.

## Common mistakes at this boundary

### Mistake 1: assuming TCP is message-oriented

Tests on a quiet machine often produce one read per write, teaching an accidental contract. Load, buffering, offload, scheduling, and receiver call sizes break the coincidence. Always frame a stream explicitly and test fragmented and coalesced input.

### Mistake 2: assuming UDP drops are always in the network

UDP datagrams can be discarded before the application consumes them because the receive queue is full. The application can truncate a received datagram with an undersized buffer. Code can reject a source or sequence number. Observe kernel counters, queue state, `MSG_TRUNC`, and application discard counters before attributing every missing message to the path.

### Mistake 3: treating non-blocking mode as automatic concurrency

`O_NONBLOCK` only changes how calls report inability to progress. The program still needs a readiness mechanism, fair scheduling, state ownership, bounded buffers, deadlines, and error handling. A busy loop repeatedly calling `recv` until data appears is non-blocking at the syscall level and blocking at the CPU level.

### Mistake 4: equating TCP acknowledgement with processing

An ACK confirms transport receipt into the peer's TCP machinery, subject to the protocol's rules. It does not confirm that a handler parsed, authorized, committed, or replicated the request. Applications needing that guarantee must return an application-level result and define retry behavior for ambiguous outcomes.

### Mistake 5: copying a "safe UDP size" without stating the path

There is no single maximum payload that is optimal for every path. A conservative fixed size can avoid fragmentation on many paths but waste capacity elsewhere. A large fixed size can fail through tunnels. Established protocols combine an initial bound with PMTU discovery and fallback behavior. Custom protocols must do the same or explicitly constrain their deployment environment.

## Run it yourself

### Question

Does changing only the socket type change the receiver-visible boundary from an ordered byte stream to individual datagrams, even when the application makes the same three logical sends?

### Preconditions

Use the Linux `netlab` created in [the series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The namespaces must be named `c` and `s`, with client address `10.77.0.1` on `c0` and server address `10.77.0.2` on `s0`. This experiment requires root or capabilities to enter network namespaces. It requires Python 3, `ip`, and `ss` from iproute2. On macOS, run it inside the privileged Linux VM used for the series lab because network namespaces are Linux facilities.

The experiment mutates no qdisc, route, firewall, or sysctl. It creates one temporary script, starts short-lived processes on debug port 9090, and removes only that script during reset.

Run this preflight first:

```bash
set -euo pipefail
ip netns list | awk '{print $1}' | grep -Fx c
ip netns list | awk '{print $1}' | grep -Fx s
ip -n c -brief address show dev c0
ip -n s -brief address show dev s0
ip -n c route get 10.77.0.2
ip netns exec c python3 --version
ip netns exec s ss -V
ip netns exec s ss -Hlnpt 'sport = :9090' || true
```

Read the interface lines and the route result. Expected: `c0` includes `10.77.0.1/30`, `s0` includes `10.77.0.2/30`, and the route to `10.77.0.2` uses `c0`. The final `ss` command should print no row before the experiment. If it prints a row, another process owns port 9090; stop and choose a free debug port consistently in every command.

Create the single lab program:

```bash
cat >/tmp/socket-contract-lab.py <<'PY'
#!/usr/bin/env python3
import argparse
import socket
import time

MESSAGES = [b"alpha", b"bravo", b"charlie"]
SERVER = ("10.77.0.2", 9090)

parser = argparse.ArgumentParser()
parser.add_argument("role", choices=("server", "client"))
parser.add_argument("transport", choices=("tcp", "udp"))
args = parser.parse_args()

kind = socket.SOCK_STREAM if args.transport == "tcp" else socket.SOCK_DGRAM

if args.role == "server":
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.bind(SERVER)
        if args.transport == "tcp":
            sock.listen(1)
            print("READY tcp", flush=True)
            conn, peer = sock.accept()
            with conn:
                chunks = []
                while True:
                    chunk = conn.recv(4)
                    if not chunk:
                        break
                    chunks.append(chunk)
                    print(f"TCP_RECV bytes={len(chunk)} data={chunk!r}", flush=True)
                joined = b"".join(chunks)
                print(f"TCP_TOTAL calls={len(chunks)} bytes={len(joined)} data={joined!r}", flush=True)
        else:
            print("READY udp", flush=True)
            total = 0
            for index in range(len(MESSAGES)):
                payload, peer = sock.recvfrom(65535)
                total += len(payload)
                print(f"UDP_RECV index={index} bytes={len(payload)} data={payload!r}", flush=True)
            print(f"UDP_TOTAL calls={len(MESSAGES)} bytes={total}", flush=True)
else:
    with socket.socket(socket.AF_INET, kind) as sock:
        sock.connect(SERVER)
        for payload in MESSAGES:
            sock.sendall(payload)
            print(f"SEND bytes={len(payload)} data={payload!r}", flush=True)
        if args.transport == "tcp":
            sock.shutdown(socket.SHUT_WR)
        time.sleep(0.2)
PY
chmod 0755 /tmp/socket-contract-lab.py
```

The program uses the same three payloads in both modes. The receive operation is intentionally different because the contracts are different: TCP requests at most 4 stream bytes per call, while UDP provides a buffer large enough for one whole lab datagram.

### Baseline

Run the TCP baseline:

```bash
set -euo pipefail
tcp_log=/tmp/socket-contract-tcp.log
ip netns exec s python3 /tmp/socket-contract-lab.py server tcp >"$tcp_log" 2>&1 &
tcp_server_pid=$!
for attempt in $(seq 1 50); do
  grep -q '^READY tcp$' "$tcp_log" && break
  sleep 0.05
done
grep -q '^READY tcp$' "$tcp_log"
ip netns exec s ss -lntp 'sport = :9090'
ip netns exec c python3 /tmp/socket-contract-lab.py client tcp
wait "$tcp_server_pid"
sed -n '/^TCP_/p' "$tcp_log"
```

Read the `ss` header and listener row before the client runs. Confirm `Netid` is TCP, `State` is `LISTEN`, the local endpoint is `10.77.0.2:9090`, and the process metadata names Python when permissions expose it. Then read `TCP_RECV` and `TCP_TOTAL`.

Expected: the concatenated `TCP_TOTAL` data is `b'alphabravocharlie'`, exactly 17 bytes. Because the server asks for at most 4 bytes per receive, it needs at least 5 calls and at most 17 calls. On a quiet local lab it will commonly use 5 calls, but that exact count is not a protocol guarantee. The chunks do not preserve the client's 5-byte, 5-byte, and 7-byte send boundaries.

### Apply one change

Change only the transport from TCP stream to UDP datagrams. The addresses, port, messages, and send-call count remain the same.

```bash
set -euo pipefail
udp_log=/tmp/socket-contract-udp.log
ip netns exec s python3 /tmp/socket-contract-lab.py server udp >"$udp_log" 2>&1 &
udp_server_pid=$!
for attempt in $(seq 1 50); do
  grep -q '^READY udp$' "$udp_log" && break
  sleep 0.05
done
grep -q '^READY udp$' "$udp_log"
ip netns exec s ss -lunp 'sport = :9090'
ip netns exec c python3 /tmp/socket-contract-lab.py client udp
wait "$udp_server_pid"
sed -n '/^UDP_/p' "$udp_log"
```

Read the socket row again. Confirm `Netid` is UDP and there is no TCP `LISTEN` state machine. Then inspect each `UDP_RECV` record.

Expected in this zero-impairment, directly connected namespace lab: exactly 3 receive calls with payload lengths 5, 5, and 7 bytes, totaling 17 bytes. Each call returns one sent datagram. The observed local order will normally match send order, but UDP does not promise that order on a general path. This run proves message boundaries, not Internet delivery or ordering.

### Compare

Use these exact summaries:

```bash
printf 'TCP summary: '
grep '^TCP_TOTAL' /tmp/socket-contract-tcp.log
printf 'UDP summary: '
grep '^UDP_TOTAL' /tmp/socket-contract-udp.log
printf 'TCP receive calls: '
grep -c '^TCP_RECV' /tmp/socket-contract-tcp.log
printf 'UDP receive calls: '
grep -c '^UDP_RECV' /tmp/socket-contract-udp.log
```

Read the `calls` and `bytes` fields. Both modes should deliver 17 total bytes in this no-loss lab. UDP should report exactly 3 receive calls because each receive consumes one datagram. TCP should report 5–17 calls with this 4-byte receive buffer, independent of the client's 3 logical sends. Scheduler timing can change the TCP partition, but it cannot restore a message-boundary promise that TCP does not make.

This comparison connects directly to the post's main claim. The socket API shape is similar, but the selected transport changes the receiver-visible contract. Application framing is mandatory on the stream. Loss, order, congestion, and size policy remain application protocol responsibilities on UDP.

### Reset

Remove only the processes and temporary files created by this experiment:

```bash
set -euo pipefail
pkill -f '/tmp/socket-contract-lab.py' 2>/dev/null || true
rm -f /tmp/socket-contract-lab.py \
  /tmp/socket-contract-tcp.log \
  /tmp/socket-contract-udp.log
ip netns exec s ss -Hlnpt 'sport = :9090' || true
ip netns exec s ss -Hlunp 'sport = :9090' || true
```

Expected: both final `ss` commands print no rows. The namespace topology, interfaces, routes, qdiscs, and the series `netserver` remain unchanged.

### Production translation

On a production Linux host, use read-only commands and narrow filters. Do not enter an arbitrary namespace or capture payloads without authorization.

```bash
sudo ss -ntpi state established '( sport = :8080 or dport = :8080 )'
sudo ss -lnpt 'sport = :8080'
sudo ss -uanp 'sport = :9090'
uname -r
ss -V
```

Take at least two snapshots several seconds apart before calling a queue "growing." Record the namespace, kernel version, iproute2 version, protocol, endpoint tuple, and process owner. If packet evidence is necessary, use a targeted filter, bounded duration, and suitable snap length. Captures can expose sensitive application data.

## Key takeaways

- A socket is the local boundary between application intent and kernel-managed protocol state. A successful syscall is evidence about that boundary, not proof of remote application processing.
- TCP is an ordered reliable byte stream. It does not preserve application write boundaries, so every stream protocol needs explicit framing and bounded frame sizes.
- UDP preserves each delivered datagram boundary but does not promise delivery, order, duplicate suppression, congestion control, or a usable path size.
- Blocking and non-blocking modes change how local waiting is scheduled. They do not change the transport contract. Non-blocking correctness requires explicit state, bounded buffers, readiness discipline, and deadlines.
- A production protocol over UDP must rebuild or deliberately omit ordering, reliability, congestion response, PMTU handling, security, and peer validation. QUIC demonstrates both the value and the cost.
- Read `ss` from left to right: protocol, state, queues, endpoints, owner, then protocol internals. Interpret every field according to the socket type and state.
- Diagnose outward from the boundary. Application state and socket queues come before packet captures; packet captures come before broad speculation about the network.

## Further reading

- [RFC 9293: Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293.html), August 2022
- [RFC 8085: UDP Usage Guidelines](https://www.rfc-editor.org/rfc/rfc8085.html), March 2017
- [RFC 8899: Datagram Packetization Layer Path MTU Discovery](https://www.rfc-editor.org/rfc/rfc8899.html), September 2020
- [RFC 9000: QUIC, a UDP-Based Multiplexed and Secure Transport](https://www.rfc-editor.org/rfc/rfc9000.html), May 2021
- [Linux socket(7), tcp(7), udp(7), epoll(7), and ss(8) manuals](https://man7.org/linux/man-pages/man7/socket.7.html)
- [How a packet gets there: ARP, switching, routing, and ECMP](/blog/software-development/networking/how-a-packet-gets-there-arp-switching-routing-and-ecmp)
- [The latency budget: speed of light, serialization, and queueing](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing)
