---
title: "Nagle, Delayed ACKs, and the 40 ms Mystery"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to prove a Nagle and delayed-ACK stall from packet timing, repair the message boundary, and choose TCP_NODELAY or TCP_CORK without guessing."
tags:
  [
    "networking",
    "distributed-systems",
    "tcp",
    "nagle-algorithm",
    "delayed-ack",
    "tcp-nodelay",
    "tcp-cork",
    "rpc-latency",
    "packet-capture",
    "linux",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/nagle-delayed-acks-and-the-40ms-mystery-1.webp"
---

A service is fast for most requests, yet one cluster in its latency histogram sits near 40 ms. The client and server share a host or rack. CPU is idle. There is no loss, no retransmission, and no slow database query. The trace contains a stranger clue: the client sends a small piece of the request, pauses, then sends the rest almost exactly when an ACK arrives.

That shape is not proof by itself, but it is a valuable fingerprint. It can arise when the sender applies Nagle's small-segment rule, the receiver applies delayed acknowledgments, and the application splits one logical request across multiple small writes before reading the response. Each layer makes a locally sensible decision. Together they put the request behind an acknowledgment timer.

The first diagram is the mental model: the lost time can sit inside the request phase of a warm connection, before the server has the complete frame and before its handler can do useful work.

![A warm RPC latency ladder with the request phase highlighted at a 40 ms lab signature](/imgs/blogs/nagle-delayed-acks-and-the-40ms-mystery-1.webp)

This article builds the diagnosis from bytes and packets. We will derive the wait, distinguish it from retransmission and scheduling, reproduce it in the series `netlab`, and choose among coalescing writes, `TCP_NODELAY`, `TCP_CORK`, and pipelining. The 40 ms value is a Linux implementation clue, not a TCP guarantee. Other stacks and connection histories can produce a different delay or no delay at all.

If you need the full journey around this one TCP edge, start with [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The neighboring posts on [the TCP handshake and its cost](/blog/software-development/networking/the-tcp-handshake-and-what-it-costs-you) and [reliability, ACKs, retransmits, and RTO](/blog/software-development/networking/reliability-sequence-numbers-acks-retransmits-and-rto) own connection establishment and loss recovery. Here we assume an established, lossless connection and ask why correct bytes can still leave late.

## 1. The 40 ms fingerprint

**Senior rule: a repeated round number is usually a timer, a batch boundary, or a retry policy until the trace proves otherwise.**

A database can take 40 ms. A scheduler can also leave a process off CPU for 40 ms. Neither explanation naturally produces a packet sequence in which a small outbound segment follows an ACK with microsecond-scale consistency. That causal adjacency is the useful clue.

Suppose a warm RPC has these illustrative components:

| Component | Time | Classification | Source |
| --- | ---: | --- | --- |
| DNS | 0 RTT | Existing connection | Derived from the warm-connection assumption |
| TCP handshake | 0 RTT | Existing connection | Derived from the warm-connection assumption |
| TLS handshake | 0 RTT | Existing session | Derived from the warm-connection assumption |
| Request completion | expected 35–60 ms | Delayed-ACK timer candidate | Reproducible range for the `netlab` treatment below on supporting Linux kernels |
| Server work | Not isolated | Outside this experiment's claim | Not measured |
| First-byte flight | Not isolated | Outside this experiment's claim | Not measured |
| Response transfer | Not isolated | Outside this experiment's claim | Not measured |

The table does not claim that all Linux RPCs spend 40 ms here. It asks us to place the symptom at the correct boundary. If the server capture shows only the first request fragment during the gap, the handler cannot yet own that time. If the client capture shows the second fragment already on the wire, Nagle is not holding it. If an ACK arrived promptly but the process did not run, look at scheduling instead.

### Turn a histogram into a packet question

A useful incident question is not, "Why is p99 40 ms?" It is:

> For a request in the slow mode, which endpoint had the next byte needed for forward progress, and what event released it?

That question forces a finite set of answers:

- The application had not called `write` yet. Instrument application timestamps and scheduler delay.
- The application called `write`, but the sender's TCP stack retained the bytes. Inspect socket options and the packet capture.
- The sender transmitted the bytes, but the path lost or queued them. Inspect sequence numbers, retransmissions, and both capture points.
- The receiver had the bytes but had not delivered or processed them. Inspect receive queues, framing, runtime pauses, and handler timing.
- The receiver owed only an ACK, and its delayed-ACK policy had not sent it yet. Inspect the ACK timestamp and the segment that immediately follows it.

This is why a span named `client.send` is often insufficient. A successful `write(2)` means the kernel accepted bytes into a socket buffer. It does not mean the peer received them, or even that a segment has left the host. For this bug class, packet timing is the missing observability boundary. Higher-level tracing still matters, but it must be joined to wire evidence. [Observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) covers that broader instrumentation problem.

### Why the peak is bimodal

The fast requests take a path where the request is already one sendable unit, an ACK arrives before the second write, a second segment triggers an immediate ACK, or `TCP_NODELAY` permits the next short segment. The slow requests take a path where one short segment remains unacknowledged while another short write waits behind it. A timer is discrete, so it adds a separate mode more readily than a smooth tail.

The simplified latency model below is explanatory, not an equation stated by TCP:

$$
T_{rpc} = T_{app} + T_{path} + I_{stall} T_{ack}.
$$

Here, $T_{app}$ is application work, $T_{path}$ is the ordinary request and response flight time, $T_{ack}$ is the receiver's effective delayed-ACK wait, and $I_{stall}$ is 1 only when the write pattern and TCP state trigger the interaction. If $T_{app} + T_{path}$ is 2 ms and $T_{ack}$ is 40 ms in this lab, the two modes appear near 2 ms and 42 ms. That arithmetic is derived from the model. It is not a promise about a production kernel.

The bimodality can disappear under load. More packets provide more ACK opportunities, batching changes write boundaries, and connections spend less time idle. A load test may therefore look healthy while sparse production traffic stalls. Lower concurrency can expose the bug more clearly because each request has fewer neighboring packets to break the wait.

### A 40 ms mode is a lead, not a verdict

Linux's current TCP header defines `TCP_DELACK_MIN` as `HZ/25` when `HZ` is at least 100. Since `HZ` is ticks per second, the expression represents $1/25$ second, or 40 ms. The same source defines a larger delayed-ACK ceiling, and the actual algorithm adapts using connection state and RTT. See the current [Linux TCP constants](https://github.com/torvalds/linux/blob/master/include/net/tcp.h) and ACK scheduling logic in [`tcp_output.c`](https://github.com/torvalds/linux/blob/master/net/ipv4/tcp_output.c).

That is enough to explain why 40 ms is recognizable on Linux. It is not enough to diagnose a trace by numerology. [RFC 9293, Section 3.8.6.3](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6.3) requires delayed ACKs to remain below 0.5 seconds and recommends an ACK for at least every second full-sized segment or `${2}` times the receiver MSS. The standard does not require a 40 ms timer. Kernel version, RTT estimate, ACK ratio, quick-ACK state, traffic direction, offload, and application behavior all affect what appears.

## 2. Two optimizations form one stall

**Senior rule: never call either side broken until you can state the local rule each side is following.**

![Packet timeline showing Nagle holding a second write until the delayed ACK timer releases the request](/imgs/blogs/nagle-delayed-acks-and-the-40ms-mystery-2.webp)

Nagle's algorithm is a sender-side coalescing rule. Delayed ACK is a receiver-side acknowledgment rule. They are independent. The failure requires a particular application dependency across them.

### Nagle's rule, precisely enough to debug

John Nagle's January 1984 [RFC 896](https://www.rfc-editor.org/rfc/rfc896.html) addressed the small-packet problem. A one-byte Telnet payload over IPv4 and TCP could carry 40 bytes of headers, so applications that wrote a character at a time could create far more packet work than useful data. The RFC's original context matters. Nagle was not trying to optimize a modern RPC median. He was protecting a shared network from a sender that emitted tiny increments.

The current TCP specification states the operational rule in [RFC 9293, Section 3.7.4](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.7.4): when data is already unacknowledged, buffer new user data until the outstanding data is acknowledged or enough new data exists for a full-sized segment. It also requires a per-connection way for applications to disable the algorithm.

For diagnosis, track three sender facts:

1. `SND.UNA` is the oldest unacknowledged sequence number.
2. `SND.NXT` is the next sequence number the sender will use.
3. The new pending bytes are smaller than the effective send MSS.

If `SND.NXT > SND.UNA`, some data is outstanding. A second small write can remain queued even though `write(2)` returned successfully. An ACK that advances `SND.UNA` can then release it. A packet capture exposes this without requiring direct access to kernel variables: one short data segment goes out, a gap follows, an ACK arrives, and the next short data segment immediately follows.

Nagle does not mean "wait 40 ms." It owns no 40 ms timer in this story. It waits for either an ACK or enough queued data. The receiver's ACK policy supplies the visible timer.

### Delayed ACK, precisely enough to debug

A pure ACK consumes bandwidth and packet-processing work even though it carries no application payload. A receiver can wait briefly after one data segment in case any of three useful events occurs:

- another segment arrives, allowing one cumulative ACK to cover both;
- the local application produces response data, allowing the ACK to ride with it;
- the receive window changes, allowing one packet to carry both the ACK and window update.

[RFC 1122, Section 4.2.3.2](https://www.rfc-editor.org/rfc/inline-errata/rfc1122.html) standardized the broad behavior in October 1989: a TCP should implement delayed ACK, the delay must be less than 0.5 seconds, and a stream of full-sized segments should receive an ACK at least every second segment. [RFC 9293](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6.3) carries the modern requirement.

Delayed ACK does not mean every ACK waits. Implementations can acknowledge immediately for out-of-order data, a second segment, connection startup, protocol heuristics, or an explicitly requested quick-ACK phase. Linux's `TCP_QUICKACK` is not a permanent off switch. The [`tcp(7)` manual](https://man7.org/linux/man-pages/man7/tcp.7.html) says the stack can later enter or leave quick-ACK mode according to normal protocol processing.

That detail explains many failed reproductions. A developer runs one request on a fresh loopback connection, sees no stall, and concludes the mechanism is obsolete. The first packets may be in quick-ACK state. The relevant question is whether the affected long-lived connection entered delayed-ACK behavior before a split request.

### The dependency cycle

Consider a length-prefixed RPC whose client performs `write(header)`, `write(body)`, then `read(response)`. Assume both writes are smaller than the MSS and the connection has no other traffic:

1. The first write leaves immediately because there is no prior unacknowledged application data.
2. The server receives one short segment. It can legally delay its ACK because it expects another segment or response data.
3. The client performs the second write. Nagle sees unacknowledged data and a new short write, so it holds those bytes.
4. The server's framing layer waits for the body before it calls the handler. It therefore has no response to piggyback with the ACK.
5. The client application waits for the response. It does not produce more bytes that would fill an MSS.
6. The receiver's delayed-ACK timer expires and sends a pure ACK.
7. The ACK advances the client's acknowledged sequence. Nagle releases the body.
8. The server receives a complete frame, runs the handler, and responds.

There is no protocol deadlock in the permanent sense because the timer provides an escape. Operationally, however, a synchronous RPC can be rate-limited by the timer. A fixed 40 ms serial wait implies an upper bound of approximately

$$
\frac{1\ \text{second}}{0.040\ \text{second/request}} = 25\ \text{requests/second}
$$

per connection before server work and ordinary network time. The 25 requests per second figure is derived from a 40 ms wait, not measured throughput. Parallel connections or multiplexed requests can hide it in aggregate throughput while each affected RPC remains slow.

### It is not really a deadlock between algorithms

Engineers often say Nagle and delayed ACK "deadlock each other." The phrase is memorable, but the actual dependency includes the application protocol. Nagle waits for an ACK or a full segment. Delayed ACK waits for another segment, response data, or its timer. The server application waits for the rest of the framed request. The client application waits for the response.

Remove any dependency and progress resumes:

- Coalesce the header and body into one send operation.
- Make the second write large enough to fill a segment.
- Disable Nagle for this latency-sensitive connection.
- Send another segment or another pipelined request.
- Have the receiver ACK immediately, though that is usually the least portable application-level control.
- Change the framing so the server can act on the first fragment, if the protocol semantics allow streaming.

The application boundary is therefore the best repair point. Treating delayed ACK as the villain and globally changing receiver behavior can increase ACK traffic for every workload on the host while preserving the split-write bug in code.

## 3. Watch the write-write-read collision

**Senior rule: a timing bug becomes understandable when every wait has an owner and a release event.**

<figure class="blog-anim">
<svg viewBox="0 0 880 360" role="img" aria-label="Header crosses to the server, the body waits behind Nagle until delayed ACK fires, then the body and response complete" style="width:100%;height:auto;max-width:880px">
<style>
.nd40-lane{stroke:var(--border,#d1d5db);stroke-width:2}.nd40-node{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.nd40-text{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.nd40-note{font:500 13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.nd40-packet{fill:var(--accent,#6366f1)}.nd40-block{fill:var(--accent,#6366f1);opacity:.65}.nd40-ok{fill:var(--accent,#6366f1)}.nd40-ack{stroke:var(--accent,#6366f1);stroke-width:3;stroke-dasharray:7 5;fill:none}.nd40-hidden{opacity:0}
@keyframes nd40-header{0%,8%{transform:translateX(0);opacity:1}22%,100%{transform:translateX(540px);opacity:1}}
@keyframes nd40-body{0%,48%{transform:translateX(0);opacity:1}62%,100%{transform:translateX(540px);opacity:1}}
@keyframes nd40-timer{0%,24%{stroke-dashoffset:100;opacity:0}28%{opacity:1}50%{stroke-dashoffset:0;opacity:1}55%,100%{opacity:0}}
@keyframes nd40-response{0%,66%{transform:translateX(0);opacity:0}72%{opacity:1}92%,100%{transform:translateX(-540px);opacity:1}}
@keyframes nd40-status{0%,48%{opacity:1}55%,100%{opacity:0}}
.nd40-header{animation:nd40-header 11s ease-in-out infinite}.nd40-body{animation:nd40-body 11s ease-in-out infinite}.nd40-timer{animation:nd40-timer 11s ease-in-out infinite}.nd40-response{animation:nd40-response 11s ease-in-out infinite}.nd40-status{animation:nd40-status 11s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.nd40-header,.nd40-body,.nd40-timer,.nd40-response,.nd40-status{animation:none}.nd40-header{transform:translateX(540px)}.nd40-body{transform:translateX(540px)}.nd40-response{transform:translateX(-540px);opacity:1}.nd40-status{opacity:0}.nd40-timer{opacity:1;stroke-dashoffset:0}}
</style>
<rect class="nd40-node" x="40" y="42" width="180" height="58" rx="10"/><rect class="nd40-node" x="660" y="42" width="180" height="58" rx="10"/>
<text class="nd40-text" x="130" y="77">client application</text><text class="nd40-text" x="750" y="77">server application</text>
<line class="nd40-lane" x1="130" y1="108" x2="130" y2="326"/><line class="nd40-lane" x1="750" y1="108" x2="750" y2="326"/>
<g class="nd40-header"><rect class="nd40-packet" x="150" y="128" width="110" height="34" rx="7"/><text class="nd40-text" x="205" y="151" style="fill:#fff">header</text></g>
<g class="nd40-body"><rect class="nd40-block" x="150" y="184" width="110" height="34" rx="7"/><text class="nd40-text" x="205" y="207" style="fill:#fff">body</text></g>
<text class="nd40-note nd40-status" x="350" y="238">Nagle holds body while bytes remain unacknowledged</text>
<path class="nd40-ack nd40-timer" d="M750 252 H130"/><text class="nd40-note" x="440" y="274">delayed ACK timer fires near 40 ms in this lab trace</text>
<g class="nd40-response"><rect class="nd40-ok" x="620" y="292" width="110" height="34" rx="7"/><text class="nd40-text" x="675" y="315" style="fill:#fff">response</text></g>
</svg>
<figcaption>Header flight starts the collision: the body waits behind Nagle, delayed ACK releases it, and the completed frame finally produces a response.</figcaption>
</figure>

The animation follows the causal order, not a claim that every connection uses an identical timer. The first request fragment is real data on the wire. The second fragment exists only in the sender's queue. The server cannot form the logical request, so it cannot answer. The delayed-ACK timer is the only changing state until it fires.

### Why `write-write-read` is the dangerous shape

TCP is a byte stream. It does not preserve the application's calls to `write`. Two writes may become one segment, and one write may become several segments. That freedom is essential, but it means an API call boundary is not a message boundary.

The dangerous shape has four properties:

- one logical request is split into at least two small writes;
- the first write becomes visible as a short TCP segment;
- the second write arrives at the TCP sender while the first remains unacknowledged;
- the protocol stops producing data until a response arrives.

Any missing property can eliminate the stall. A buffered writer may coalesce both pieces. TLS may place them in one record or may split them differently. A larger body may fill an MSS and escape Nagle's hold. A bidirectional protocol may have unrelated traffic that causes a prompt ACK. A fresh connection may still be in quick-ACK mode.

This sensitivity is why a harmless-looking refactor can create or remove the problem. Replacing one `writev` call with separate serialization and body writes changes packetization even though the byte stream is identical. Adding tracing can change allocation and flush timing. Upgrading a runtime can change the default socket option. A production-only histogram can therefore originate in a boundary far above the socket API.

### Write boundaries through TLS and RPC layers

TLS adds its own record framing between the application and TCP. It does not restore TCP message boundaries. A library may emit a record header and payload together, buffer them, or use multiple underlying writes. The exact behavior depends on the library and version, so do not infer it from source-level calls alone.

Likewise, an RPC stack may serialize a fixed header, compression metadata, and payload through different buffers. gRPC and HTTP/2 have their own framing and multiplexing semantics, so the right place to learn those contracts is [gRPC and Protocol Buffers](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming). The diagnostic principle remains: capture what TCP emitted, then walk upward until you find the flush or write boundary that produced it.

On encrypted traffic, packet capture still shows segment lengths, flags, ACK numbers, and timing even when payload contents are opaque. You usually do not need decryption to prove the stall. You need to know that the second data segment was absent until an ACK arrived.

### Why local tests are especially deceptive

Loopback has tiny RTT, no physical loss, and large effective MSS. Those facts should make RPCs fast, but they can also sharpen the timer signature. Ordinary propagation contributes almost nothing, so a 40 ms policy wait dominates the request.

At the same time, loopback offloads and quick-ACK heuristics can combine or acknowledge traffic differently from a veth path. Capturing on `lo` may show aggregated views that differ from packets on a physical NIC. Use the `c` and `s` namespaces from `netlab`, capture both `c0` and `s0` when necessary, and treat offload-visible segment sizes carefully. The series introduction explains the [stable netlab topology](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url).

## 4. Choose the fix at the message boundary

**Senior rule: repair the producer's message boundary before reaching for a host-wide TCP knob.**

![Matrix comparing write coalescing, TCP_NODELAY, TCP_CORK, and protocol pipelining](/imgs/blogs/nagle-delayed-acks-and-the-40ms-mystery-3.webp)

The popular advice is to set `TCP_NODELAY`. It is often correct for latency-sensitive RPCs, but it answers only one question: may TCP send a short segment while earlier data remains unacknowledged? It does not make a fragmented serializer efficient, create application framing, flush a user-space buffer, or make an overloaded event loop run.

Start with the logical message. Decide when the peer has enough bytes to act, then make the transport policy express that decision.

| Strategy | What it controls | Good fit | Main risk |
| --- | --- | --- | --- |
| One buffer or `writev` | Application write boundary | Header and small body known together | Copy cost if implemented with unnecessary concatenation |
| `TCP_NODELAY` | Sender's Nagle behavior | Interactive RPC, control messages, latency-sensitive duplex traffic | Many tiny segments if the application writes carelessly |
| `TCP_CORK` | Explicit Linux flush boundary | Header plus `sendfile`, or assembled bulk response | Forgotten uncork delays output; not portable |
| Pipelining or multiplexing | Protocol dependency | Multiple independent requests per connection | More in-flight state and harder cancellation |
| `TCP_QUICKACK` | Receiver ACK behavior, temporarily | Controlled experiments and specialized stacks | Linux-specific hint, not persistent, easy to misuse |

### First choice: one logical request, one gathering write

If the header and body are already available, a gathering write communicates intent cleanly. `writev(2)` supplies multiple buffers to one system call without requiring an extra concatenation buffer. The kernel still owns packetization, but it can see the entire logical unit at once instead of reacting to two separately timed calls.

This C fragment is a real shape, with error handling shortened only around retry policy:

```c
#include <errno.h>
#include <stdint.h>
#include <sys/socket.h>
#include <sys/uio.h>
#include <unistd.h>

int send_frame(int fd, const void *body, size_t body_len) {
    uint32_t be_len = htonl((uint32_t)body_len);
    struct iovec parts[2] = {
        {.iov_base = &be_len, .iov_len = sizeof(be_len)},
        {.iov_base = (void *)body, .iov_len = body_len},
    };

    size_t wanted = sizeof(be_len) + body_len;
    ssize_t sent = writev(fd, parts, 2);
    if (sent < 0)
        return -errno;
    if ((size_t)sent != wanted)
        return -EAGAIN; /* production code queues the unsent suffix */
    return 0;
}
```

Production code must handle partial writes, `EINTR`, and nonblocking `EAGAIN` without duplicating bytes. A short `writev` does not invalidate the technique. It means the connection's output queue must remember which iovec and offset remain.

Coalescing is not equivalent to "copy everything into one giant buffer." `writev`, `sendmsg`, language runtime gathering APIs, and buffered encoders can preserve zero-copy or low-copy behavior. The goal is to expose an intentional flush boundary.

There is also a deeper benefit: one message boundary makes metrics meaningful. You can time `frame_ready`, `enqueue`, `write_complete`, and `response_first_byte` rather than timing two unrelated serializer calls. That separation helps distinguish user-space backpressure from kernel-side coalescing.

### `TCP_NODELAY`: low latency, not automatic quality

On Linux, `TCP_NODELAY` disables Nagle so segments are sent as soon as possible even when only a small amount of data is ready. The authoritative operational description is in [`tcp(7)`](https://man7.org/linux/man-pages/man7/tcp.7.html). Set it per accepted or connected socket, not as a vague host tuning exercise.

```c
#include <netinet/tcp.h>
#include <sys/socket.h>

int enable_nodelay(int fd) {
    int one = 1;
    return setsockopt(fd, IPPROTO_TCP, TCP_NODELAY, &one, sizeof(one));
}
```

The option is a good default when each small write represents data the peer can use immediately and latency matters more than minimizing packet count. Interactive shells, game inputs, consensus control messages, and request headers followed by a small body often fit.

It is a poor excuse for emitting one byte per function call in a tight loop. With IPv4 and no TCP options, one byte of payload can require a 20-byte IPv4 header and a 20-byte TCP header. The wire overhead ratio is therefore

$$
\frac{20 + 20}{1} = 40,
$$

or 40 header bytes per payload byte. This is a derived minimum example. Ethernet framing, VLAN tags, TCP timestamps, tunnels, and encryption can add more overhead. Offloads may reduce host CPU work, but they do not make a pathological application boundary a good protocol design.

Check your runtime before adding redundant configuration. Go's current [`net.TCPConn.SetNoDelay`](https://go.dev/src/net/tcpsock.go) documentation says the default is `true`, which means Go TCP connections send after a `Write` without Nagle delay. A Go service that already has this default can still suffer user-space buffering, TLS record behavior, proxy-side Nagle, or a different peer implementation. Never infer the entire path from one endpoint's language.

### `TCP_CORK`: hold deliberately, then flush deliberately

`TCP_CORK` solves the opposite coordination problem on Linux. While corked, TCP holds partial frames so the application can prepend a header, append file data, and then release a more complete segment. Clearing the option flushes queued partial data. The Linux manual documents a 200 ms ceiling for corked output and notes that `TCP_CORK` has been combinable with `TCP_NODELAY` since Linux 2.5.71.

The classic shape is an HTTP server sending a generated header and a file:

```c
#include <netinet/tcp.h>
#include <sys/sendfile.h>
#include <sys/socket.h>
#include <unistd.h>

int send_header_and_file(int fd, const char *header, size_t header_len,
                         int file_fd, off_t *offset, size_t file_len) {
    int one = 1;
    int zero = 0;
    if (setsockopt(fd, IPPROTO_TCP, TCP_CORK, &one, sizeof(one)) < 0)
        return -1;
    if (send(fd, header, header_len, MSG_NOSIGNAL) < 0)
        goto fail;
    if (sendfile(fd, file_fd, offset, file_len) < 0)
        goto fail;
    return setsockopt(fd, IPPROTO_TCP, TCP_CORK, &zero, sizeof(zero));

fail:
    setsockopt(fd, IPPROTO_TCP, TCP_CORK, &zero, sizeof(zero));
    return -1;
}
```

The important line is the cleanup path. A cork is state. If an error, cancellation, or early return skips the uncork, latency becomes the cork ceiling rather than the intended flush time. Wrap it in a scope guard or `defer` in languages that support one, and add a test that aborts between header and body.

`TCP_CORK` is Linux-specific and is generally a better server-side bulk assembly tool than a portable RPC default. Some platforms expose related mechanisms with different semantics. If portability matters, start with a buffered writer or gathering write whose flush boundary is explicit in application code.

### `TCP_NODELAY` and `TCP_CORK` are not opposites you toggle casually

Linux gives cork precedence over no-delay while the cork remains set. Setting `TCP_NODELAY` can force an explicit flush of pending output even when corked, as documented by `tcp(7)`. That interaction is useful in carefully designed libraries and confusing in layered stacks where two components believe they own transport policy.

Assign one owner. The connection constructor should establish baseline options. The framing or file-transfer layer may temporarily cork if it can guarantee uncorking. A serializer should not flip socket flags per field. Otherwise, a header library, TLS layer, proxy library, and application can fight over the same state with no trace at the API boundary.

Record the chosen option once per connection in structured diagnostics. On Linux, `ss -tin` may show `nodelay` or `cork` in its internal TCP information, depending on kernel and iproute2 version. Treat absence from display cautiously and verify with `getsockopt` in the owning process when possible.

### Pipelining changes the dependency, not the timer

If the protocol permits multiple requests in flight, pipelining prevents one response from becoming the only release event. Stuart Cheshire's case study recommends keeping another request behind the one being awaited so the packet pipeline does not go empty. Multiplexed protocols generalize the idea: independent work can continue even when one stream waits.

Pipelining is not a free transport switch. It increases in-flight state, complicates cancellation, and can move head-of-line blocking into the application protocol. A server must bound queued work and responses. A client must correlate responses. The design belongs alongside backpressure, admission control, and failure handling, not inside an isolated socket helper.

For a simple request/response protocol, coalescing the request and setting an appropriate fixed socket policy are usually safer than retrofitting pipelining. For a high-throughput multiplexed protocol, the empty-pipeline dependency may indicate a broader design limitation.

### Do not make `TCP_QUICKACK` the production fix

Linux exposes `TCP_QUICKACK` to request immediate rather than delayed acknowledgments. It is tempting because it appears to attack the visible timer. The manual is explicit that the flag is not permanent. Subsequent protocol processing can re-enter or leave quick-ACK mode.

That makes it useful in a controlled lab. It is less attractive as the only correctness or latency guarantee of an application protocol. It is also not portable. More importantly, the receiver is often not the component you own. Fixing the sender's message boundary and per-connection Nagle choice travels with the application.

Use quick ACK as evidence: if forcing immediate ACKs removes the mode, you have isolated ACK timing as a necessary condition. Then repair the sender or protocol so progress does not depend on a receiver-specific hint.

### The second-order cost: packet rate

Disabling coalescing can trade latency for packet rate. The cost shows up in interrupt moderation, softirq work, per-packet firewall and conntrack processing, proxy bookkeeping, and NIC queues. Bytes per second can stay flat while packets per second rise sharply.

Measure both outcomes:

```bash
# Read-only commands on a Linux endpoint.
ss -tin dst 10.77.0.2:8080
nstat -az 'Tcp*'
ip -s link show dev c0
ethtool -S c0 2>/dev/null | grep -Ei 'packet|drop|miss' || true
```

Capture request latency, data segments per request, average payload bytes per data segment, retransmissions, and CPU time in the network stack. A fix that removes a 40 ms tail at the cost of changing two segments into three is usually an excellent trade for RPC control traffic. A fix that turns a bulk stream into thousands of tiny segments needs application coalescing.

## 5. The 2005 WiFi conformance test

**Senior rule: when one byte changes throughput by almost a factor of two, inspect segment arithmetic before blaming the device.**

![Before and after comparison of the 2005 WiFi test around a one-byte payload boundary](/imgs/blogs/nagle-delayed-acks-and-the-40ms-mystery-4.webp)

Stuart Cheshire published a concrete case on May 20, 2005 in [TCP Performance Problems Caused by Interaction between Nagle's Algorithm and Delayed ACK](https://www.stuartcheshire.org/papers/nagledelayedack/). The case was a WiFi conformance testing program, not an anonymous production anecdote. It repeatedly sent a fixed data block over TCP and waited for an application-level acknowledgment.

The observed result looked like an operating-system or radio-performance problem. Windows met the test's 3.5 Mb/s threshold. Mac OS X transferred the 100,000-byte blocks at 2.7 Mb/s and failed. Engineers then changed the buffer size by tiny amounts. A 99,912-byte buffer reached 5.2 Mb/s, while 99,913 bytes fell back to 2.7 Mb/s.

Those figures are reported by Cheshire for that specific program, platform behavior, and test in 2005. They are not a current macOS or Windows benchmark.

| Test condition | Reported result | Interpretation | Source |
| --- | ---: | --- | --- |
| Required WiFi test threshold | 3.5 Mb/s | Pass criterion | Stuart Cheshire, May 20, 2005 |
| Mac OS X, 99,912-byte buffer | 5.2 Mb/s | Passed because segment parity avoided the pause | Stuart Cheshire, May 20, 2005 |
| Mac OS X, 99,913-byte buffer | 2.7 Mb/s | Failed after one byte changed the tail behavior | Stuart Cheshire, May 20, 2005 |
| Mac OS X, 100,000-byte buffer | 2.7 Mb/s | Reported failing case with repeated pauses | Stuart Cheshire, May 20, 2005 |

### Derive the boundary rather than memorizing it

The Mac OS X trace used a 1,448-byte TCP MSS because TCP timestamps consumed 12 bytes relative to the 1,460-byte MSS in the compared Windows path. Cheshire decomposed 100,000 bytes as

$$
100{,}000 = 69 \times 1{,}448 + 88.
$$

The multiplication gives 99,912 bytes, leaving an 88-byte tail. Sixty-nine is odd. The receiver promptly acknowledged pairs of full-sized segments through segment 68, then received segment 69 alone. Delayed ACK held that acknowledgment. Nagle held the 88-byte tail because unacknowledged data remained. The receiver application could not produce its one-byte application acknowledgment because it still lacked the tail.

The useful comparison is one byte above 99,912. The exact packetization details in the reported implementation create a boundary where the final short data and ACK cadence interact badly. A one-byte source-level change did not change radio capacity. It changed which side of a transport threshold the request occupied.

The pauses in Cheshire's trace were about 200 ms, not 40 ms. That distinction matters. The mechanism transfers across systems; the timer value does not. The Linux 40 ms signature comes from a common minimum delayed-ACK interval in current Linux source. The 2005 case demonstrates the same dependency with the observed timer and stacks of that environment.

### Evidence ledger

The case clears the series evidence gate:

- **Case:** WiFi conformance testing program described by Stuart Cheshire.
- **Event date:** published May 20, 2005; Cheshire also says he first encountered the broader problem in 1999.
- **Source:** the direct public engineering write-up linked above.
- **Source owner:** Stuart Cheshire, who described his own testing encounter and Apple experience.
- **Mechanism:** odd full-sized segment count, short tail held by Nagle, receiver ACK held by delayed ACK, application response waiting for the complete block.
- **Verified numbers:** 3.5 Mb/s requirement, 2.7 Mb/s failing result, 5.2 Mb/s passing result, 99,912 and 99,913 byte boundary observations, 1,448-byte MSS, and repeated roughly 200 ms pauses.
- **Transfer lesson:** suspicious performance cliffs around payload-size boundaries demand packet and MSS arithmetic, not a generic platform label.

### Trigger, contributing conditions, and multiplier

The trigger was the block size and resulting final segment pattern. The contributing conditions were a stop-and-wait application protocol, Nagle at the sender, delayed ACK at the receiver, and a response that could not be generated until the full block arrived. The blast-radius multiplier was repetition: every block could encounter another timer wait.

That decomposition prevents weak lessons. "Disable Nagle on Mac" would overfit one historical implementation. "Windows networking was faster" would be false for the mechanism. "Make the buffer 99,912 bytes" would encode an MSS-dependent accident. The transferable controls are to keep useful work in flight, coalesce semantic messages deliberately, set a per-connection latency policy, and verify segment arithmetic on the deployed path.

### A second public debugging account

Julia Evans described another 40 ms case in her March 13, 2017 SREcon keynote transcript, [So You Want to Be a Wizard](https://jvns.ca/blog/so-you-want-to-be-a-wizard/). A Ruby HTTP client and server were on the same computer. Packet capture showed headers leaving, a 40 ms gap, then the rest of the request. She attributed the behavior to Nagle and delayed ACK, and reported that setting `TCP_NODELAY` removed the wait.

That account is valuable because it demonstrates the diagnostic move: the server looked slow from the request duration, yet Wireshark placed the gap on the client's outbound request. It also gives the derived throughput intuition that a synchronous 40 ms wait limits one connection to 25 request cycles per second before other costs.

Neither case authorizes claiming every 40 ms pause is this interaction. Together they show the same causal shape in two different contexts and timer regimes. The packet sequence, not the folklore, is what makes the diagnosis transferable.

## 6. Why the RPC histogram becomes bimodal

**Senior rule: treat the histogram as a map of execution paths, not as a single noisy distribution.**

![Conceptual histogram with a fast mode and a delayed ACK timer mode near 40 ms](/imgs/blogs/nagle-delayed-acks-and-the-40ms-mystery-5.webp)

The figure is a conceptual model. Its 0–5 ms fast band and 35–50 ms delayed band are the expected ranges for the controlled Linux lab below, not reported production measurements. Validate any real service with its own request IDs and packet capture.

A histogram mixes requests that may have taken different network-state paths. For this interaction, the hidden variables include:

- whether the connection was new, idle, or continuously busy;
- whether quick-ACK state was active;
- whether the serializer flushed after the header;
- whether TLS or a buffered writer coalesced the pieces;
- whether the first fragment remained unacknowledged at the second write;
- whether another segment arrived and triggered an ACK;
- whether `TCP_NODELAY` was set on that exact socket;
- whether a proxy opened a second connection with different defaults.

If requests randomly land across two connection populations, the mixture can be approximated as

$$
p(T) = (1-q)p_{fast}(T) + q\,p_{stall}(T),
$$

where $q$ is the fraction that takes the stalled path. This is an explanatory mixture model, not a TCP formula. If `p_fast` centers near 2 ms and the stall adds about 40 ms, `p_stall` centers near 42 ms. Changing $q$ changes the height of the second mode without moving it much. Changing the ACK timer or receiver stack moves the mode.

That distinction guides experiments. If enabling `TCP_NODELAY` collapses the slow mode into the fast mode, the mechanism likely changed. If the slow mode stays at the same location but becomes rarer, connection routing or write coalescing may have changed. If the entire distribution shifts, look at path RTT or server work.

### Do not hide modes behind percentiles

A single p99 number cannot tell whether 1 percent of requests each waited on the same 40 ms timer or whether all requests experienced a continuous range of queueing. Export a latency histogram with buckets that can resolve the suspected interval, and retain exemplars or request IDs when the metrics system supports them.

The following Prometheus-style bucket set is illustrative. It is chosen to separate a sub-5 ms mode from a 35–50 ms mode:

```yaml
rpc_client_request_seconds:
  type: histogram
  buckets:
    - 0.001
    - 0.002
    - 0.005
    - 0.010
    - 0.020
    - 0.030
    - 0.035
    - 0.040
    - 0.045
    - 0.050
    - 0.075
    - 0.100
```

Those are configuration choices, not measurements. Coarse default buckets such as 10 ms, 50 ms, and 100 ms can conceal the exact shape. Do not respond by creating hundreds of high-cardinality labels. Keep dimensions bounded, and connect a sampled request ID to packet capture or detailed logs during a controlled investigation.

### Join application time to capture time

Packet captures use host clocks and may be collected at different points. A one-sided capture is often enough because the crucial ordering occurs on one sender: data segment, gap, ACK arrival, next data segment. For two-sided captures, synchronize clocks or compare relative intervals rather than assuming absolute timestamps align.

At the application, log the monotonic timestamps `frame_started`, `header_encoded`, `body_encoded`, `first_write_returned`, `second_write_returned`, `read_started`, and `first_response_byte` for a sampled request.

This is a field list, not fabricated output. Compare `second_write_returned` with the capture carefully. The call can return before transmission. A better advanced signal is a kernel event or `TCP_INFO` snapshot, but those add platform coupling. Start with application events plus `tcpdump`, then add deeper instrumentation only if the boundary remains ambiguous.

### The proxy can create a second mode

A client may set `TCP_NODELAY` correctly while an L7 proxy uses a different default on its upstream connection. The service histogram then splits by route, proxy instance, or connection age. Capture at both sides of the proxy or compare its downstream and upstream connection metrics.

Do not collapse this into "the network." Name the specific TCP leg. An end-to-end RPC over client, proxy, sidecar, and server may contain three independent TCP connections, each with its own Nagle setting, ACK behavior, RTT, MSS, and write boundaries. The [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) treats each termination as a new transport contract.

### Low load can be the worst load

At high throughput, full-sized segments and frequent ACKs keep the sender moving. At low throughput, a lone short segment has no follower. That makes the delayed-ACK timer visible. Autoscaling can create an odd symptom: adding capacity reduces per-connection traffic and increases the fraction of sparse connections that hit the stall.

Before declaring a capacity regression, stratify by requests per connection, connection idle time, payload size, and proxy hop. A latency regression after scaling out can be a packet-cadence change rather than resource contention. This is also why synthetic health checks, which often create a new connection or send a different payload, can miss the affected state.

## 7. A packet-first diagnostic runbook

**Senior rule: prove where the missing bytes waited before changing a socket option.**

The shortest reliable investigation starts with one slow request and one fast request on the same TCP leg. Compare packet order, not just total duration.

### Step 1: localize the incomplete request

Capture only the relevant flow. On a lab namespace, this is safe and bounded:

```bash
sudo ip netns exec c timeout 15 \
  tcpdump -i c0 -nn -tt -s 128 -c 200 \
  'tcp and host 10.77.0.2 and port 8080' \
  -w /tmp/nagle-client.pcap

tshark -r /tmp/nagle-client.pcap \
  -Y 'tcp.port == 8080' \
  -T fields \
  -e frame.time_relative \
  -e ip.src -e tcp.srcport -e ip.dst -e tcp.dstport \
  -e tcp.seq -e tcp.ack -e tcp.len -e tcp.flags
```

Read `frame.time_relative`, `tcp.seq`, `tcp.ack`, and `tcp.len`. For the suspected interaction, the slow request has this qualitative sequence:

1. a short client data segment;
2. no retransmission and no second client data segment during the gap;
3. a pure server ACK near the end of the gap;
4. the next short client data segment immediately after that ACK;
5. the response only after the server receives the second segment.

This is not ASCII packet art. It is a list of properties to verify in the capture.

If both request fragments reached the server before the gap, leave this runbook. Measure server scheduling, framing, handler work, and response buffering. If the first fragment was retransmitted, use the loss and RTO workflow from [reliability, sequence numbers, ACKs, retransmits, and RTO](/blog/software-development/networking/reliability-sequence-numbers-acks-retransmits-and-rto). If a zero window appears, investigate receive-side flow control in [two windows, one pipe](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe).

### Step 2: prove that the gap follows the ACK timer

Calculate two intervals from packet timestamps:

$$
\Delta_{ack} = t_{ack} - t_{first\_data}
$$

and


$$
\Delta_{release} = t_{second\_data} - t_{ack}.
$$

These are explanatory measurements. If $\Delta_{ack}$ clusters around 35–50 ms in the Linux lab and $\Delta_{release}$ stays below roughly 1 ms on an idle host, ACK arrival is the release event. If the second segment follows 10 ms after the ACK, application scheduling or a user-space buffer may still own the gap.

Use `tshark` to extract data and pure ACK frames separately:

```bash
# Client data segments.
tshark -r /tmp/nagle-client.pcap \
  -Y 'ip.src == 10.77.0.1 && tcp.dstport == 8080 && tcp.len > 0' \
  -T fields -e frame.number -e frame.time_relative \
  -e tcp.seq -e tcp.len -e tcp.analysis.push_bytes_sent

# Server pure ACKs.
tshark -r /tmp/nagle-client.pcap \
  -Y 'ip.src == 10.77.0.2 && tcp.srcport == 8080 && tcp.len == 0 && tcp.flags.ack == 1' \
  -T fields -e frame.number -e frame.time_relative -e tcp.ack
```

The optional `tcp.analysis.push_bytes_sent` field depends on Wireshark version and analysis state. Sequence, length, ACK number, and relative time are the durable fields.

### Step 3: inspect the exact socket

Do not inspect a listener and assume accepted sockets inherited every intended behavior. Query the established connection while the experiment runs:

```bash
sudo ip netns exec c ss -tin dst 10.77.0.2:8080
sudo ip netns exec s ss -tin src 10.77.0.2:8080
```

Read the local and peer ports first so you know both commands refer to the same connection. Then inspect `nodelay`, `cork`, `rtt`, `ato`, `bytes_sent`, `bytes_acked`, `segs_out`, and `segs_in` when your iproute2 build exposes them. `ato` is ACK timeout state, not a stable user-facing delayed-ACK configuration value.

In application code, log a `getsockopt(TCP_NODELAY)` result once after connection establishment. Avoid per-request calls just for telemetry. If a proxy terminates TCP, inspect both downstream and upstream accepted sockets.

### Step 4: change one condition

The best discriminating changes are:

- same byte stream, one `writev` instead of two `write` calls;
- same two calls, `TCP_NODELAY` off versus on at the sender;
- same sender, delayed ACK forced for a lab receiver versus normal behavior;
- same socket options, payload large enough to fill an MSS versus a short payload.

Do not combine all four. If a deploy enables `TCP_NODELAY`, changes buffering, upgrades TLS, and increases concurrency, a faster histogram does not identify the mechanism.

### Step 5: count the cost of the fix

After latency improves, compare data-segment count and CPU:

```bash
tshark -r /tmp/nagle-client.pcap \
  -Y 'ip.src == 10.77.0.1 && tcp.dstport == 8080 && tcp.len > 0' \
  -T fields -e tcp.stream -e tcp.len |
awk '{segments[$1]++; bytes[$1]+=$2}
     END {for (s in segments) print s, segments[s], bytes[s]}' |
sort -n
```

The output columns are stream number, data segment count, and payload bytes seen in the capture. This is not a universal cost model because offload and capture location can change the visible segmentation. It is still a good before-and-after check in the same namespace and capture point.

### Common false positives

**Retransmission timeout:** an original sequence range appears again, often after a much larger timer. Nagle does not retransmit the first fragment.

**Receive-window stall:** the receiver advertises a zero or tiny window. The sender waits for a window update, not merely an ACK that advances `SND.UNA`.

**User-space buffering:** the packet does not appear after `TCP_NODELAY`, and the application has not flushed a buffered writer or TLS stream. The bytes never reached the TCP decision point.

**Scheduler or garbage-collection pause:** an ACK arrives, but the next write system call or response processing occurs much later. Join runtime pause telemetry to monotonic timestamps.

**Proxy response buffering:** the upstream server responds promptly, but an intermediary waits to fill or flush its downstream buffer. Compare both TCP legs.

**Application timer:** a retry, batching window, or event-loop tick happens to be 40 ms. Packet order will show that no ACK causally releases the next segment.

### Decision record for a safe production change

Before enabling `TCP_NODELAY` broadly, write down:

1. the affected connection role and library;
2. the packet sequence that proves bytes waited in sender TCP;
3. the controlled test that removes the mode;
4. segment-count and CPU change;
5. rollout metric and rollback threshold;
6. whether any proxy or sidecar owns another TCP leg;
7. the long-term plan to coalesce semantic messages.

This turns a folklore flag into an auditable transport decision.

![Decision tree for distinguishing Nagle and delayed ACK from server work, loss, and scheduling](/imgs/blogs/nagle-delayed-acks-and-the-40ms-mystery-6.webp)

## 8. Design the RPC so the timer cannot own progress

**Senior rule: a request protocol should not require a transport ACK before it can finish sending one small logical request.**

The safest protocol sends a complete small request from already available data, using one gathering operation or an explicit buffered flush. When bodies stream, the protocol should make that streaming state explicit rather than pretending multiple fragments are an atomic request.

### Length prefixes and flush contracts

A length prefix helps only when the announced bytes follow promptly. In Go, `net.Buffers.WriteTo` can gather a header and payload without an extra concatenation copy, while the connection constructor retains Go's default no-delay behavior. Document that the framing call returns after the connection writer accepts the bytes, not after the peer receives them. A test transport can record the number and timing of underlying writes.

Avoid APIs where any helper can flush the shared writer. A metrics header, compression prelude, or authentication field should contribute bytes to the frame builder, not secretly turn itself into a packet boundary.

### Timeouts must exceed more than the server budget

If a client timeout is 30 ms and the receiver's effective ACK delay can be 40 ms, the request can fail before the server sees the complete frame. Retrying may open a new connection that succeeds due to quick ACK, creating a misleading recovery pattern.

Do not solve this by blindly raising timeouts. First remove the transport dependency. Then choose an end-to-end deadline from the service budget. The SRE-side retry and backoff policy belongs in [timeouts, retries, and backoff done right](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right). This article owns only the wire mechanism that can consume that budget before handler execution.

### Backpressure and cancellation

Coalescing does not mean buffering without bounds. A frame builder should cap message size, respect context cancellation, and expose output-queue backpressure. If a body is huge, stream it with a protocol designed for chunks. If a body is small, do not split it merely because the encoder has two objects.

When `writev` returns partially on a nonblocking socket, retain the unsent suffix in order. Cancellation must not allow the next request to reuse a connection whose previous frame is only partially written unless the protocol can resynchronize. Often the safe choice is to close that connection.

### Test invariants, not packet counts

Exact packet counts vary with MSS, TCP options, offloads, TLS records, and kernel implementation. A portable integration test should assert:

- a small frame completes without waiting on an ACK-only timer;
- enabling or disabling Nagle according to policy is observable on the socket;
- the write layer exposes one semantic flush for one small frame;
- cancellation cannot leave a partial frame followed by another request;
- segment rate remains within an operational budget under tiny-message load.

Use packet-count assertions only in a controlled Linux netlab with offloads and topology stated. Production correctness must not depend on 1,448 bytes forever.

## Run it yourself

### Question

Can two small client writes on one established TCP connection produce a delayed mode when Nagle is enabled and the receiver enters delayed-ACK mode, while the same byte stream completes promptly with `TCP_NODELAY`?

This experiment forces the ACK-policy precondition so the result is teachable. It does not claim that every ordinary Linux connection will choose delayed ACK for the same packets.

### Preconditions

Use the Linux-only `netlab` topology from [the series introduction](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url): namespaces `c` and `s`, interfaces `c0` and `s0`, addresses `10.77.0.1` and `10.77.0.2`, and server port `8080`. You need root or capabilities for network namespaces and packet capture, plus Python 3, `tcpdump`, `tshark`, `ss`, and GNU `timeout`.

The experiment writes two temporary Python files under `/tmp`, starts one process in namespace `s`, and captures only port 8080 on `c0`. Packet captures can contain application payloads, tokens, and personal data. Use the narrow filter shown here and do not copy the command to an unspecified production interface.

Confirm the environment before changing anything:

```bash
set -euo pipefail
sudo ip netns list | grep -E '^(c|s)([[:space:]]|$)'
sudo ip -n c addr show dev c0
sudo ip -n s addr show dev s0
sudo ip -n c route get 10.77.0.2
sudo ip -n s route get 10.77.0.1
sudo ip netns exec s ss -lnt 'sport = :8080'
python3 --version
tcpdump --version | head -n 1
tshark --version | head -n 1
```

Expected state: both namespaces and veth interfaces exist, each route resolves through its named interface, and port 8080 is initially unused. If `ss` prints a listener, stop here and choose a free lab port consistently instead of killing an unknown process.

Create the receiver. `TCP_QUICKACK=0` asks Linux to leave quick-ACK mode before each frame header. The option is a lab control, is Linux-specific, and can be overridden by later TCP processing.

```bash
cat >/tmp/nd40-server.py <<'PY'
import socket

ADDR = ("10.77.0.2", 8080)
QUICKACK = getattr(socket, "TCP_QUICKACK", 12)

def read_exact(conn, size):
    out = bytearray()
    while len(out) < size:
        chunk = conn.recv(size - len(out))
        if not chunk:
            raise EOFError
        out.extend(chunk)
    return bytes(out)

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(ADDR)
    listener.listen(1)
    conn, peer = listener.accept()
    with conn:
        for _ in range(42):
            conn.setsockopt(socket.IPPROTO_TCP, QUICKACK, 0)
            header = read_exact(conn, 4)
            length = int.from_bytes(header, "big")
            body = read_exact(conn, length)
            if body != b"abcdefgh":
                raise ValueError(body)
            conn.sendall(b"!")
PY
```

Create the client. The first two iterations warm connection state and are excluded. The 40 recorded iterations print `latency_ms`, which is the field to compare. The environment variable `NODELAY` is the only treatment switch.

```bash
cat >/tmp/nd40-client.py <<'PY'
import os
import socket
import statistics
import time

ADDR = ("10.77.0.2", 8080)
NODELAY = int(os.environ.get("NODELAY", "0"))
samples = []

with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as conn:
    conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, NODELAY)
    conn.connect(ADDR)
    for index in range(42):
        started = time.monotonic_ns()
        conn.sendall((8).to_bytes(4, "big"))
        time.sleep(0.001)
        conn.sendall(b"abcdefgh")
        if conn.recv(1) != b"!":
            raise RuntimeError("bad response")
        elapsed = (time.monotonic_ns() - started) / 1_000_000
        if index >= 2:
            samples.append(elapsed)
            print(f"request={index - 1:02d} latency_ms={elapsed:.3f}")

ordered = sorted(samples)
print(
    "summary",
    f"n={len(samples)}",
    f"min_ms={ordered[0]:.3f}",
    f"median_ms={statistics.median(ordered):.3f}",
    f"p95_ms={ordered[int(0.95 * (len(ordered) - 1))]:.3f}",
    f"max_ms={ordered[-1]:.3f}",
)
PY
```

### Baseline

Start the receiver, begin a bounded capture, then run the client with Nagle enabled:

```bash
set -euo pipefail
sudo ip netns exec s python3 /tmp/nd40-server.py &
SERVER_PID=$!
sudo ip netns exec c timeout 20 tcpdump -i c0 -nn -s 128 \
  'tcp and host 10.77.0.2 and port 8080' \
  -w /tmp/nd40-nagle.pcap >/tmp/nd40-tcpdump.log 2>&1 &
CAPTURE_PID=$!
sleep 0.5
sudo ip netns exec c env NODELAY=0 python3 /tmp/nd40-client.py |
  tee /tmp/nd40-nagle.txt
wait "$SERVER_PID"
sudo kill -INT "$CAPTURE_PID" 2>/dev/null || true
wait "$CAPTURE_PID" 2>/dev/null || true
```

Read the final `summary` line, especially `median_ms` and `p95_ms`. Then inspect `frame.time_relative`, `tcp.seq`, `tcp.ack`, and `tcp.len`:

```bash
tshark -r /tmp/nd40-nagle.pcap -Y 'tcp.port == 8080' \
  -T fields -e frame.time_relative -e ip.src -e ip.dst \
  -e tcp.seq -e tcp.ack -e tcp.len -e tcp.flags |
  sed -n '1,80p'
```

Expected on a typical Linux kernel that honors the forced delayed-ACK state: most recorded baseline requests land roughly in the 35–60 ms range, with a short client segment, an ACK near the end of the gap, and the second client data segment immediately after it. Scheduler load, kernel version, timer granularity, veth offloads, and virtualization can widen the range.

If baseline remains below 10 ms, do not fabricate a 40 ms result. Preserve the capture, record `uname -a`, and report that this kernel's ACK heuristics defeated the forced precondition. The experiment is still falsifiable. The packet sequence should show which event differed.

### Apply one change

The treatment changes only the client socket's `TCP_NODELAY` value from 0 to 1. The two writes, 1 ms application gap, receiver, namespace path, payload, and request count remain the same.

```bash
set -euo pipefail
sudo ip netns exec s python3 /tmp/nd40-server.py &
SERVER_PID=$!
sudo ip netns exec c timeout 20 tcpdump -i c0 -nn -s 128 \
  'tcp and host 10.77.0.2 and port 8080' \
  -w /tmp/nd40-nodelay.pcap >/tmp/nd40-tcpdump.log 2>&1 &
CAPTURE_PID=$!
sleep 0.5
sudo ip netns exec c env NODELAY=1 python3 /tmp/nd40-client.py |
  tee /tmp/nd40-nodelay.txt
wait "$SERVER_PID"
sudo kill -INT "$CAPTURE_PID" 2>/dev/null || true
wait "$CAPTURE_PID" 2>/dev/null || true
```

### Compare

```bash
printf '%s\n' 'Nagle enabled:'
tail -n 1 /tmp/nd40-nagle.txt
printf '%s\n' 'TCP_NODELAY enabled:'
tail -n 1 /tmp/nd40-nodelay.txt

for capture in /tmp/nd40-nagle.pcap /tmp/nd40-nodelay.pcap; do
  printf 'capture=%s client_data_segments=' "$capture"
  tshark -r "$capture" \
    -Y 'ip.src == 10.77.0.1 && tcp.dstport == 8080 && tcp.len > 0' \
    -T fields -e frame.number | wc -l
done
```

Read the `median_ms` and `p95_ms` fields. Expected treatment latency is roughly 0.5–8 ms in this namespace lab because it includes the explicit 1 ms sleep, Python scheduling, and two local flights. The treatment should no longer wait for the 35–60 ms ACK-timer band. Expect at least two client data-bearing segments per request because the script deliberately separates the writes; offloads and capture placement can change what is visible.

Connect the observation to the claim: the byte stream and receiver did not change. Allowing the second short segment to leave while the first was unacknowledged removed the ACK timer from the request's critical path.

For a production-safe read-only translation, capture a bounded affected flow, inspect `ss -tin` for the exact socket, and compare a canary connection whose application explicitly sets `TCP_NODELAY`. Do not set `TCP_QUICKACK`, alter qdiscs, or capture broad payload traffic on a production host without an approved plan.

### Reset

Remove only this experiment's processes and temporary files:

```bash
sudo pkill -f '^python3 /tmp/nd40-server.py$' 2>/dev/null || true
rm -f /tmp/nd40-server.py /tmp/nd40-client.py \
  /tmp/nd40-nagle.pcap /tmp/nd40-nodelay.pcap \
  /tmp/nd40-nagle.txt /tmp/nd40-nodelay.txt \
  /tmp/nd40-tcpdump.log
```

The reset does not delete namespaces, interfaces, routes, qdiscs, or unrelated processes. It leaves the canonical `netlab` ready for the next post.

## Key takeaways

- Nagle and delayed ACK are independent optimizations. The stall appears only when the application's framing and stop-and-wait dependency connect them.
- A 40 ms mode is a Linux-flavored timer clue, not a TCP constant. Other implementations can show a different timer, and ACK heuristics can avoid the wait entirely.
- The proving packet sequence is short data, gap, pure ACK, then the next short data immediately after the ACK, with no retransmission in between.
- Fix the message boundary first. Use one buffer, `writev`, or a disciplined flush so one small logical request does not depend on a transport ACK to finish leaving.
- `TCP_NODELAY` is usually right for latency-sensitive small messages, but measure the resulting segment rate and CPU. It does not repair user-space buffering.
- `TCP_CORK` is a Linux-specific assembly tool for deliberate batching, especially headers plus bulk data. It requires a guaranteed uncork path.
- `TCP_QUICKACK` is a temporary Linux hint and a useful experimental control. It is a weak foundation for an application protocol.
- When a one-byte payload change creates a large throughput cliff, compute MSS and segment parity before blaming the network device or operating system.
- Preserve the distinction between a warm connection's request phase and server work. A server cannot process bytes it has not received.
- The safest production change comes with packet proof, one controlled treatment, packet-rate accounting, a canary, and a rollback threshold.

## Further reading

- [RFC 896: Congestion Control in IP/TCP Internetworks](https://www.rfc-editor.org/rfc/rfc896.html), John Nagle, January 1984.
- [RFC 1122: Requirements for Internet Hosts, Communication Layers](https://www.rfc-editor.org/rfc/inline-errata/rfc1122.html), October 1989, especially Section 4.2.3.2.
- [RFC 9293: Transmission Control Protocol](https://www.rfc-editor.org/rfc/rfc9293.html), August 2022, especially Sections 3.7.4 and 3.8.6.3.
- [`tcp(7)` Linux manual page](https://man7.org/linux/man-pages/man7/tcp.7.html) for `TCP_NODELAY`, `TCP_CORK`, `TCP_QUICKACK`, and `TCP_INFO`.
- [TCP Performance Problems Caused by Interaction between Nagle's Algorithm and Delayed ACK](https://www.stuartcheshire.org/papers/nagledelayedack/), Stuart Cheshire, May 20, 2005.
- [So You Want to Be a Wizard](https://jvns.ca/blog/so-you-want-to-be-a-wizard/), Julia Evans, March 13, 2017, for the debugging story behind a 40 ms local HTTP gap.
- [API payload size, compression, and tail latency](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) for the higher-level design boundary around serialization and payload trade-offs.
