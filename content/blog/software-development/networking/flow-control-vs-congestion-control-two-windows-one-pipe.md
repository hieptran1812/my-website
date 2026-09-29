---
title: "Flow Control vs Congestion Control: Two Windows, One Pipe"
date: "2026-09-29"
publishDate: "2026-09-29"
description: "Learn to separate TCP receiver pressure from path congestion, calculate the window a long fast path needs, and tune Linux without guessing."
tags:
  [
    "networking",
    "distributed-systems",
    "tcp",
    "flow-control",
    "congestion-control",
    "linux",
    "socket-buffers",
    "bandwidth-delay-product",
    "performance",
    "observability",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/flow-control-vs-congestion-control-two-windows-one-pipe-1.webp"
---

A team buys a 10 Gbit/s circuit between two regions, moves a large file over a single TCP connection, and gets about 40 Mbit/s. The first theory is congestion. Someone changes the congestion-control algorithm. Nothing useful happens. Someone else blames encryption, disk, or the cloud provider. The link is almost empty throughout.

The missing question is not “How fast is the pipe?” It is “How many bytes may this sender keep in the pipe before it must wait?” TCP answers with two independent limits. The receiver advertises `rwnd`, the amount of sequence space it is prepared to accept. The sender maintains `cwnd`, the amount it believes the network can safely carry. New data in flight is bounded by the smaller limit, plus the amount the application can actually supply.

![The end-to-end path with the TCP transfer and its two-window gate highlighted](/imgs/blogs/flow-control-vs-congestion-control-two-windows-one-pipe-1.webp)

The diagram above is the mental model: one pipe, two governors, and one sender that must obey both. A low `rwnd` is receiver backpressure. A low `cwnd` is a path-capacity judgment made by congestion control. They can produce the same graph in an application dashboard, yet demand different fixes.

This post follows the packet and the socket memory behind both windows. We will derive the 40 Mbit/s ceiling, examine zero-window probes, distinguish Linux autotuning from `SO_RCVBUF`, and build a runbook around `ss`, `iperf3`, and a bounded capture. For the wider request path, start with [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). For the retransmission machinery behind loss recovery, keep [sequence numbers, ACKs, retransmits, and RTO](/blog/software-development/networking/reliability-sequence-numbers-acks-retransmits-and-rto) nearby.

## The symptom lives in the transfer phase

The connection succeeds quickly. The TLS handshake succeeds. The server starts sending immediately. Then the transfer settles at a disappointing, stable rate. That stability tempts people to call the rate “available bandwidth.” It may instead be the most boring arithmetic in TCP.

As an explanatory upper-bound model, let $W$ be the permitted bytes in flight and $R$ be round-trip time in seconds. Ignoring protocol overhead, application stalls, and loss, one flow cannot sustain more than:

$$
T_{window} \approx \frac{8W}{R}
$$

The sender's usable flight allowance is approximately:

$$
W \approx \min(rwnd, cwnd)
$$

These are explanatory bounds, not equations mandated by the TCP specification. They are useful because a sender that may transmit only $W$ bytes must wait for acknowledgments to advance the left edge of its send window. Each replenishment takes information traveling across an RTT.

Suppose the effective window is 750,000 bytes and RTT is 0.150 seconds:

$$
T_{window} \approx \frac{8 \times 750{,}000\ \text{bytes}}{0.150\ \text{s}} = 40{,}000{,}000\ \text{bit/s}
$$

That is exactly 40 Mbit/s in decimal units. The physical link can be 10 Gbit/s and remain 99.6 percent idle because the sender exhausts 750,000 bytes of permission, then waits. The link rate is not the active constraint.

This distinction matters operationally. A congestion-control change can increase or reshape `cwnd`; it cannot make the peer advertise more receive space. A larger receiver buffer can raise `rwnd`; it cannot repair loss, a queue, or a sender whose congestion window has collapsed. TCP uses a minimum because it must protect both resources at once.

> A fast interface tells you how quickly bytes could leave. The windows tell you whether TCP is allowed to keep supplying those bytes.

The link rate still matters. It establishes the bandwidth-delay product, the amount of data that must be in flight to keep the path continuously busy. But link speed alone never proves a single flow can reach it.

## Two windows, two owners

![The receiver and sender independently compute rwnd and cwnd before the minimum gate](/imgs/blogs/flow-control-vs-congestion-control-two-windows-one-pipe-2.webp)

**Rule of thumb: identify the owner of a limit before changing it.** `rwnd` belongs to the receiving endpoint. `cwnd` belongs to the sending endpoint. Both affect the sender, but they arrive through different evidence and respond to different events.

### `rwnd` protects the receiver

TCP is a byte stream. Bytes can arrive from the network faster than the receiving application calls `read(2)` or `recv(2)`. The kernel therefore queues accepted bytes in a receive socket buffer. The receiver advertises how much additional sequence space it is willing to accept in the Window field carried in TCP segments.

[RFC 9293, published in August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6), describes the window as the range the data receiver is prepared to accept and notes the assumption that it relates to available buffer space. As data arrives, `RCV.NXT` advances. As unread bytes accumulate, less free capacity remains. The advertised right edge tells the peer how far it may send beyond the acknowledged left edge.

This is flow control: one endpoint prevents another endpoint from overrunning its memory and application consumption rate. It is not a statement about routers, queues, or competing traffic. A receiver can advertise a small or zero window on a pristine private link.

The advertised value is not necessarily equal to the total bytes allocated to the socket. Linux needs metadata and reserves space around payload accounting. The application can also set socket options that alter the allocation path. Treat `rwnd`, receive-buffer capacity, and unread payload as related quantities, not synonyms.

### `cwnd` protects the path

The congestion window is local sender state. It is maintained by the selected congestion-control algorithm. ACK progress can allow it to grow; loss, Explicit Congestion Notification, delay signals, or the algorithm's model can constrain it. The exact behavior belongs in [CUBIC, BBR, and what changing congestion control actually does](/blog/software-development/networking/congestion-control-cubic-bbr-and-what-changing-it-actually-does). The boundary here is simpler: `cwnd` limits network flight even if the receiver has abundant space.

The TCP header does not carry a `cwnd` field. A packet capture at the receiver can see advertised windows, ACKs, sequence ranges, and retransmissions. It cannot directly read the sender's internal congestion window. On Linux, query the sender with `ss -tin` and interpret `cwnd` with the reported maximum segment size (`mss`). `cwnd` is commonly shown in segments, so an approximate byte value is `cwnd * mss`.

### The minimum is necessary, but it is not the whole sender

A more practical explanatory model is:

$$
\text{new flight} \le \min(rwnd, cwnd) - \text{unacknowledged bytes}
$$

The sender must also have data queued, pacing permission, and a functioning route. Linux may report an `app_limited` delivery sample when the application, not either TCP window, fails to provide enough bytes. A qdisc or NIC can pace or queue transmission. A receiver might advertise ample space while its storage pipeline stalls later. The minimum tells us the TCP permission ceiling, not a promise that every byte will be delivered at that ceiling.

Use the owner model to reject bad fixes quickly:

| Evidence | Likely active limiter | Owner | Useful next check | Source |
| --- | --- | --- | --- | --- |
| Advertised window approaches zero while `cwnd` remains larger | Receive flow control | Receiving host and application | Socket memory, unread queue, application read cadence | [RFC 9293 §3.8.6](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6) |
| `cwnd * mss` is smaller than peer window and retransmissions rise | Congestion or loss recovery | Sending TCP and path | `ss -tin`, retransmission counters, bounded packet capture | [Linux `ss(8)`](https://man7.org/linux/man-pages/man8/ss.8.html) |
| Both windows are large but sending pauses | Application or another local limiter | Sending process or host | `app_limited`, send queue, CPU, disk, pacing | [Linux `ss(8)`](https://man7.org/linux/man-pages/man8/ss.8.html) |
| RTT grows while goodput stops growing | Queueing near a bottleneck | Path | Queue backlog and RTT under load | Derived diagnostic model; see [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure) |

That table is a decision aid, not an automatic diagnosis. `rwnd` and `cwnd` move during a connection. Sample them over time and correlate them with bytes in flight.

## From a 16-bit field to a large receive window

The word “window” hides three representations. There is a 16-bit value in each TCP header, a scale negotiated during the handshake, and a larger internal byte range maintained by each endpoint. Debugging goes wrong when values from these layers are compared without conversion.

### The field reports permission in scaled units

The TCP Window field contains an unsigned 16-bit value. With a negotiated receive scale $S$, the effective window represented by an eligible non-SYN segment is:

$$
W_{effective} = W_{field} \times 2^S
$$

This is a protocol formula from RFC 7323. Suppose the field is 46,875 and the shift is 4. The effective advertisement is:

$$
46{,}875 \times 2^4 = 750{,}000\ \text{bytes}
$$

That conversion is why a packet capture must include the handshake. The SYN carries the Window Scale option, but the window field in a SYN itself is never scaled. If an analyzer joins a flow late, it may show a raw value while another tool reports calculated bytes. Both can be internally correct and still appear to disagree by a power of two.

The scale an endpoint sends describes windows that endpoint will advertise later. TCP is full duplex, so each direction can negotiate a different shift. Host A's scale applies to A's receive advertisements, which control payload from B to A. Host B's scale controls the opposite direction. Never copy the scale seen in one SYN onto both directions.

The maximum shift is 14. The exact largest representable scaled window is slightly less than 1 GiB because the maximum field is 65,535 rather than 65,536. The [RFC Editor's held erratum for RFC 7323](https://www.rfc-editor.org/errata/eid5585) gives the precise distinction. Operationally, “approximately 1 GiB” is enough to show that our 187.5 MB BDP is representable. It does not guarantee either endpoint allocates or advertises that much.

### Advertised window is a moving right edge

Think in sequence space, not a bucket refilled once per RTT. `RCV.NXT` is the next byte the receiver expects. `RCV.WND` covers sequence numbers beginning there that the receiver will accept. As in-order payload arrives, `RCV.NXT` advances. The receiver decides when to move the right edge based on available capacity and silly-window-syndrome avoidance.

At the sender, `SND.UNA` is the oldest unacknowledged byte, `SND.NXT` is the next sequence number it would send, and `SND.WND` is the peer's advertised allowance. RFC 9293 expresses usable window as:

$$
U = SND.UNA + SND.WND - SND.NXT
$$

That is a protocol equation from RFC 9293 §3.8.6.2.1. If $U$ is less than one effective maximum segment size, the sender generally should not drip tiny fragments onto the path merely because a few bytes opened. Sender and receiver silly-window-syndrome algorithms avoid a stable pattern of tiny window updates and tiny segments.

This also explains why “the advertised window is 8 MiB” is insufficient. If nearly 8 MiB is already unacknowledged, usable permission for new data can be close to zero. Compare the right edge with sequence state or use a tool that estimates bytes in flight.

### ACK is doing two independent jobs

One ACK segment can carry an acknowledgment number and a Window value. The acknowledgment number says which bytes arrived in order. The Window says which future bytes the receiver is prepared to accept. A receiver can acknowledge progress while reducing newly available room because the application has not consumed data quickly enough.

Congestion control reads ACK arrival as evidence about delivery, timing, and loss. Flow control reads the peer's advertised window as permission. The information travels in the same packet, but it serves different control loops. This is the source of much dashboard confusion: “ACKs are arriving” does not prove the receiver has room for another BDP of payload.

### Window shrinking is discouraged

RFC 9293 says a receiver should not move the advertised right edge left after granting sequence space. It may advertise a smaller numeric window as `RCV.NXT` advances while keeping the right edge stable, but retracting already granted space creates ambiguity. Senders must still be robust if shrinking occurs.

This matters when reading captures. A decreasing Window column is not automatically an illegal shrink. Calculate the right edge from the acknowledgment number plus the scaled window. If ACK progress advances the left edge by the same amount the numeric window falls, the receiver may simply be holding the right edge fixed while data accumulates.

| Capture observation | Correct interpretation | Source |
| --- | --- | --- |
| Window field 46,875, scale shift 4 | Effective advertisement is 750,000 bytes | Derived from [RFC 7323 §2.2](https://www.rfc-editor.org/rfc/rfc7323.html#section-2.2) |
| Window number falls while ACK number rises | Check the computed right edge before calling it window shrinking | [RFC 9293 §3.8.6](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6) |
| SYN is absent from the capture | Calculated scale may be unknown; do not trust a guessed effective window | [RFC 7323 §2.2](https://www.rfc-editor.org/rfc/rfc7323.html#section-2.2) |
| ACKs progress but effective window stays small | Delivery is occurring while receive permission remains the likely flight bound | Derived diagnostic interpretation |

## Zero window is receiver backpressure on the wire

The receiver is allowed to advertise zero. This says, “I currently accept no new payload beyond the acknowledged edge.” It does not reset the connection, prove packet loss, or mean congestion control failed.

<figure class="blog-anim">
<svg viewBox="0 0 900 360" role="img" aria-label="The TCP receive window shrinks as unread bytes fill the socket and reopens after the application drains it" style="width:100%;height:auto;max-width:900px">
<style>
.fc3-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.fc3-text{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.fc3-small{font:500 14px ui-monospace,monospace;fill:var(--text-secondary,#6b7280);text-anchor:middle}.fc3-data{fill:var(--accent,#6366f1)}.fc3-window{fill:#ffec99;stroke:#d97706;stroke-width:2}.fc3-zero{fill:#ffc9c9;stroke:#dc2626;stroke-width:2}.fc3-open{fill:#b2f2bb;stroke:#16a34a;stroke-width:2}@keyframes fc3-fill{0%,12%{transform:scaleX(.15)}42%,58%{transform:scaleX(1)}88%,100%{transform:scaleX(.15)}}@keyframes fc3-win{0%,12%{transform:scaleX(1)}42%,58%{transform:scaleX(.03)}88%,100%{transform:scaleX(1)}}@keyframes fc3-probe{0%,48%{transform:translateX(0);opacity:0}52%{opacity:1}64%{transform:translateX(370px);opacity:1}68%,100%{transform:translateX(370px);opacity:0}}@keyframes fc3-label{0%,30%,75%,100%{opacity:0}42%,65%{opacity:1}}.fc3-fill-anim{transform-origin:470px 175px;animation:fc3-fill 10s ease-in-out infinite}.fc3-win-anim{transform-origin:470px 245px;animation:fc3-win 10s ease-in-out infinite}.fc3-probe-anim{animation:fc3-probe 10s ease-in-out infinite}.fc3-zero-anim{animation:fc3-label 10s ease-in-out infinite}@media (prefers-reduced-motion:reduce){.fc3-fill-anim,.fc3-win-anim,.fc3-probe-anim,.fc3-zero-anim{animation:none}.fc3-fill-anim{transform:scaleX(.55)}.fc3-win-anim{transform:scaleX(.45)}.fc3-probe-anim{opacity:1;transform:translateX(190px)}.fc3-zero-anim{opacity:1}}
</style>
<text class="fc3-text" x="450" y="35">Receiver pressure changes rwnd, not cwnd</text>
<rect class="fc3-box" x="40" y="90" width="220" height="190" rx="14"/>
<text class="fc3-text" x="150" y="125">TCP sender</text>
<text class="fc3-small" x="150" y="155">cwnd unchanged</text>
<circle class="fc3-data fc3-probe-anim" cx="235" cy="205" r="11"/>
<text class="fc3-small" x="150" y="250">persist probe</text>
<path d="M270 205 H630" fill="none" stroke="var(--border,#d1d5db)" stroke-width="3"/>
<rect class="fc3-box" x="640" y="70" width="220" height="250" rx="14"/>
<text class="fc3-text" x="750" y="105">TCP receiver</text>
<text class="fc3-small" x="750" y="135">socket buffer</text>
<rect x="675" y="155" width="150" height="40" rx="7" fill="var(--background,#fff)" stroke="var(--border,#d1d5db)"/>
<rect class="fc3-data fc3-fill-anim" x="675" y="155" width="150" height="40" rx="7"/>
<text class="fc3-small" x="750" y="215">unread bytes grow</text>
<rect x="675" y="235" width="150" height="34" rx="7" fill="var(--background,#fff)" stroke="var(--border,#d1d5db)"/>
<rect class="fc3-window fc3-win-anim" x="675" y="235" width="150" height="34" rx="7"/>
<text class="fc3-small" x="750" y="295">app drains, rwnd reopens</text>
<g class="fc3-zero-anim"><rect class="fc3-zero" x="340" y="115" width="220" height="55" rx="10"/><text class="fc3-text" x="450" y="148">ACK, win 0</text></g>
<rect class="fc3-open" x="340" y="270" width="220" height="55" rx="10"/><text class="fc3-small" x="450" y="303">later: ACK, win &gt; 0</text>
</svg>
<figcaption>As unread bytes fill the receive socket, the advertised window closes; a persist probe keeps the connection recoverable until the application drains data and advertises space again.</figcaption>
</figure>

Motion matters here because the same connection moves through valid states. First, payload accumulates faster than the application drains it. Then free receive space disappears and the advertised window reaches zero. Later, the application consumes queued bytes and the receiver advertises a nonzero window. Congestion control need not change at any point.

### Why probes exist

Suppose the receiver sends a window update reopening the connection, but that ACK is lost. Without another event, the sender believes the window remains zero and sends nothing. The receiver waits for data. Both endpoints can wait forever.

[RFC 9293 §3.8.6.1](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6.1) closes this liveness hole. The sender must support zero-window probing. It regularly transmits at least one octet of new data, when available, or retransmits, even while the offered window is zero. A receiver that still has no space answers with an ACK carrying its current zero window. A receiver whose application has drained data can answer with the reopened value.

The RFC recommends the first probe after the zero window has existed for one retransmission timeout, then exponentially increasing the interval. It also permits the receiver to keep the window closed indefinitely. As long as the receiver keeps answering probes, the sender must allow the connection to remain open. This is the origin of the old “printer ran out of paper” example: a slow consumer can pause for an unbounded human-scale interval without converting flow control into a broken connection.

### A zero-window probe is not evidence of congestion

The probe may be retransmitted and its interval may back off, but its purpose is not to estimate bottleneck bandwidth. It tests whether the receiver's permission changed. Calling it “packet loss recovery” hides the state that matters.

In Wireshark or `tshark`, look at the advertised window from the receiving endpoint and the sequence behavior of the probe. Do not infer a zero window from application silence alone. Capture at a point where offloads do not obscure interpretation, and remember that the 16-bit header value may be scaled.

Window scaling is negotiated only during the handshake. [RFC 7323, published in September 2014](https://www.rfc-editor.org/rfc/rfc7323.html#section-2.2), defines the Window Scale option as a power-of-two shift. The option permits windows far larger than the unscaled 16-bit field, up to approximately 1 GiB with the maximum shift. If scaling was not negotiated, a later sysctl change cannot retrofit it into an established connection. Reconnect after changing buffer policy.

### Flow control can reveal an application scheduling problem

A receiver that repeatedly closes and reopens its window may simply be underprovisioned. It may also have adequate average CPU but poor read cadence. Stop-the-world pauses, event-loop blockage, thread-pool starvation, synchronous disk writes, or long work between `recv` calls let unread bytes fill the socket in bursts.

That is why increasing buffers is not automatically a complete fix. More buffer can absorb a longer scheduling gap and prevent sender stalls. It can also retain more bytes per connection, increase memory pressure across many connections, and hide an application that is not consuming promptly. The higher-level backpressure policy belongs with [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure). TCP flow control gives that policy a wire-visible symptom.

## Linux receive-buffer autotuning

![Linux automatic and explicit receive-buffer sizing follow different control paths](/imgs/blogs/flow-control-vs-congestion-control-two-windows-one-pipe-4.webp)

**Rule of thumb: inspect the application before editing a global sysctl.** Linux has an autotuning path and an explicit socket-option path. They intersect in socket memory, but their controls are not interchangeable.

### The autotuning path

The current [Linux kernel IP sysctl documentation](https://www.kernel.org/doc/html/latest/networking/ip-sysctl.html#tcp-moderate-rcvbuf) documents `net.ipv4.tcp_moderate_rcvbuf` as enabled by default. When enabled, TCP attempts to size the receive buffer for the path's throughput requirement, but no larger than the third value of `net.ipv4.tcp_rmem`.

`tcp_rmem` is a vector of three byte values:

1. `min`: memory guaranteed to a TCP socket under pressure.
2. `default`: the initial receive-buffer size.
3. `max`: the ceiling for TCP receive autotuning.

Read the live host. Do not copy a remembered default from a blog because Linux calculates some defaults from available memory and kernel lineage:

```bash
uname -r
sysctl net.ipv4.tcp_moderate_rcvbuf
sysctl net.ipv4.tcp_rmem
sysctl net.core.rmem_default
sysctl net.core.rmem_max
```

Autotuning is adaptive, not clairvoyant. A new connection begins below its eventual maximum. ACK and delivery observations give the stack evidence to grow. Memory pressure can constrain growth. A short transfer may finish before a huge target buffer matters. A connection whose application reads slowly should not advertise unlimited space merely because the path is fast.

### The explicit `SO_RCVBUF` path

An application can call `setsockopt(..., SO_RCVBUF, ...)`. The [Linux `tcp(7)` manual](https://man7.org/linux/man-pages/man7/tcp.7.html) documents two traps.

First, an explicit socket buffer request is limited by `net.core.rmem_max`. If an application asks for 256 MiB while that core limit is smaller, the requested capacity is not available through the normal unprivileged option. A privileged process can use `SO_RCVBUFFORCE`, but that is not a routine performance recommendation.

Second, Linux doubles the value requested through `SO_RCVBUF` for internal bookkeeping, and `getsockopt` therefore does not return the same number the application supplied. The `/proc` limits reflect the larger allocation rather than a one-to-one advertised payload window. Do not take a 256 MiB allocation limit, subtract nothing, and promise a 256 MiB wire window.

Explicit `SO_RCVBUF` also changes the tuning story for that socket. If a library, runtime, proxy, or benchmark tool sets it, raising only `tcp_rmem[2]` may not change the connection you care about. Find the socket call, inspect `ss -tmni`, and confirm the effective behavior.

### Which sysctl fixes the long fat path?

For an application that relies on Linux receive autotuning, the direct ceiling is `net.ipv4.tcp_rmem`'s third value. Keep `net.ipv4.tcp_moderate_rcvbuf=1`. Choose a maximum at or above the required bandwidth-delay product with room for Linux accounting and the workload's scheduling jitter. On the receiver for our 10 Gbit/s, 150 ms example, a concrete host policy is:

```ini
# /etc/sysctl.d/90-long-fat-tcp.conf
net.ipv4.tcp_moderate_rcvbuf = 1
net.ipv4.tcp_rmem = 4096 131072 268435456
```

The maximum is 256 MiB, or 268,435,456 bytes, above the 187,500,000-byte idealized BDP. This exact maximum is not universal. It is a deliberate ceiling for this stated path, and concurrent sockets multiply its memory risk.

If the application explicitly requests `SO_RCVBUF`, size `net.core.rmem_max` for the kernel's doubled accounting and set the option before `connect(2)` or `listen(2)`, as documented by `tcp(7)`. A conservative companion limit for an application requesting up to 256 MiB is:

```ini
net.core.rmem_max = 536870912
```

Better still, remove an unnecessary fixed `SO_RCVBUF` call and let autotuning operate. If a fixed value is an intentional application contract, test its returned `getsockopt` value and its wire behavior.

Do not apply these host-wide values merely because they are large. [ESnet's test and measurement host guidance, dated January 15, 2025](https://fasterdata.es.net/host-tuning/linux/test-measurement-host-tuning/), recommends a 512 MiB core maximum and a 256 MiB TCP autotuning maximum for 10 Gbit/s NICs on paths up to 200 ms RTT, or 40 Gbit/s NICs up to 50 ms RTT. ESnet explicitly scopes that profile to measurement hosts doing serialized single-stream tests. That context is the lesson: derive the path requirement, characterize concurrency, and budget memory.

### The second-order cost is per-host memory exposure

The maximum is not a reservation charged in full to every new socket, but it expands how far active sockets may grow. A host with thousands of high-bandwidth, high-RTT flows has a different safe policy from a dedicated transfer node with a few flows. `net.ipv4.tcp_mem` provides global TCP memory pressure thresholds, expressed in pages. Containers may also face cgroup memory constraints. A buffer change that wins a benchmark can make a multi-tenant service less stable.

Stage changes with four views:

```bash
# Policy
sysctl net.ipv4.tcp_moderate_rcvbuf net.ipv4.tcp_rmem net.core.rmem_max

# Per-socket memory and TCP internals
ss -tmni dst 203.0.113.10

# Global protocol counters
nstat -az | rg 'Tcp|TCPExt'

# Host memory pressure
cat /proc/net/sockstat
```

Record connection count and traffic shape. A single `iperf3` result does not price the memory tail of production concurrency.

## The 10 Gbit/s, 150 ms worked example

![Derived goodput ceilings as receive-window capacity approaches the bandwidth-delay product](/imgs/blogs/flow-control-vs-congestion-control-two-windows-one-pipe-5.webp)

**Rule of thumb: calculate bytes in flight before tuning an algorithm.** The bandwidth-delay product is the link rate multiplied by the round-trip time. It is the amount of data that can occupy the path while acknowledgments for the oldest data return.

Let link capacity $C$ be 10,000,000,000 bit/s and RTT $R$ be 0.150 s:

$$
BDP_{bits} = C \times R = 10{,}000{,}000{,}000 \times 0.150 = 1{,}500{,}000{,}000\ \text{bits}
$$

Convert to bytes:

$$
BDP_{bytes} = \frac{1{,}500{,}000{,}000}{8} = 187{,}500{,}000\ \text{bytes}
$$

That is 187.5 MB in decimal units, about 178.8 MiB in binary units. A single TCP flow needs roughly that much effective flight allowance to keep an ideal 10 Gbit/s path full at 150 ms RTT. Real systems also need headroom for accounting, ACK dynamics, application scheduling, and transient variation.

Now price several window ceilings with the explanatory bound $8W/R$:

| Effective window | Derived window-limited ceiling | Fraction of 10 Gbit/s | Source |
| --- | ---: | ---: | --- |
| 750,000 bytes | 40.0 Mbit/s | 0.40% | Derived here: $8W / 0.150$ |
| 16 MiB | 894.8 Mbit/s | 8.95% | Derived here: $8W / 0.150$ |
| 64 MiB | 3.579 Gbit/s | 35.79% | Derived here: $8W / 0.150$ |
| 187.5 MB | 10.0 Gbit/s | 100% idealized | Derived here: $8W / 0.150$ |

This table explains the classic complaint without invoking a mysterious 40 Mbit/s property of TCP. The number comes from an effective 750,000-byte window and 150 ms RTT. Change either input and the ceiling changes.

### Why parallel streams appear to fix it

If each flow is independently capped near 40 Mbit/s by its receive window, 25 similar flows have an aggregate explanatory ceiling near 1 Gbit/s, before shared path and host limits. Parallelism multiplies window state. Transfer tools historically used parallel streams to work around per-flow window constraints.

That workaround can be operationally useful, but it hides the cause. It also changes fairness, CPU cost, connection count, and failure behavior. Twenty-five flows compete differently from one. If the requirement is one large replication stream, prove one stream has adequate windows.

### Why changing CUBIC to BBR may do nothing

Assume `cwnd` has already grown beyond 187.5 MB while `rwnd` remains 750,000 bytes. The sender's effective allowance is still the minimum, 750,000 bytes. A different congestion-control algorithm can change the larger operand without changing the result.

The reverse is also true. A 256 MiB receive-buffer ceiling cannot make a sender transmit 256 MiB into a path when `cwnd` is 2 MiB. Buffers are not bandwidth. They create permission and storage for flight; congestion control still decides whether the path can carry it.

### Window scaling must already be present

The base TCP header has a 16-bit window field. Without scaling, the largest raw value is 65,535. At 150 ms, that raw ceiling implies only about 3.50 Mbit/s using the same approximation:

$$
\frac{8 \times 65{,}535}{0.150} \approx 3.50\ \text{Mbit/s}
$$

RFC 7323 scaling shifts the field value by a negotiated exponent. The scale is fixed for the connection after the SYN exchange. Verify the handshake when a supposedly large socket never advertises a large window. A middlebox or endpoint that omits the option constrains the connection regardless of later buffer growth.

## Read the evidence without confusing the windows

The fastest investigation pairs endpoint state with a packet view. Neither alone is perfect. `ss` can expose local internals such as `cwnd`; a capture can prove what the peer advertised and whether packets were lost.

### Start at the sender with `ss -tinm`

Run this repeatedly during the slow transfer:

```bash
watch -n 0.5 'ss -tinm dst 10.77.0.2'
```

The [Linux `ss(8)` manual](https://man7.org/linux/man-pages/man8/ss.8.html) defines useful fields including `rtt`, `mss`, `cwnd`, delivery rate, `rcv_space`, and socket memory. Field availability depends on kernel and iproute2 version. On the data sender, focus on:

- `cwnd` and `mss`: approximate the sender congestion allowance as their product.
- `rtt`: use the smoothed RTT to check whether the BDP assumption resembles reality.
- `bytes_acked`, delivery rate, and retransmission indicators: determine whether progress and loss fit congestion limitation.
- `send` and `pacing_rate`: distinguish observed sending from permission.
- `app_limited`, when present: treat delivery samples cautiously if the application did not fill the pipe.
- `skmem`, with `-m`: inspect allocated and configured socket memory rather than guessing from global policy.

`rcv_space` describes the local receive side, so it is not the peer's advertised receive window for the direction you are sending. This is a common analytical mistake on bidirectional connections. Always name the direction: host A's receive window controls data from B to A.

### Use a bounded capture for the peer's advertisement

Capture only the target flow and stop after a bounded time:

```bash
sudo timeout 20 tcpdump -i c0 -s 128 -w /tmp/flow-window.pcap \
  'host 10.77.0.2 and tcp port 5201'

tshark -r /tmp/flow-window.pcap \
  -Y 'tcp.port == 5201' \
  -T fields \
  -e frame.time_relative \
  -e ip.src \
  -e tcp.seq \
  -e tcp.ack \
  -e tcp.window_size_value \
  -e tcp.window_size \
  -e tcp.analysis.zero_window \
  -e tcp.analysis.window_full
```

`tcp.window_size_value` is the raw header field. `tcp.window_size` is Wireshark's calculated value after scaling when handshake context is available. Capture the handshake or scaling may be unknown. `tcp.analysis.window_full` is an analyzer inference, not a protocol flag. Use it as a lead, then inspect sequence and acknowledgment edges.

Packet captures can contain payload, credentials, tokens, and personal data. Keep the filter narrow, minimize snap length, restrict file permissions, and delete the capture according to your incident-data policy.

### Check receiver consumption

On the receiver, correlate advertised-window contraction with unread bytes and application behavior:

```bash
ss -tinm sport = :5201
pidstat -p "$(pgrep -n iperf3)" 1
cat /proc/net/sockstat
```

For a real service, add runtime-specific evidence: event-loop delay, stop-the-world pause duration, blocked-thread profiles, disk latency, or time between successful reads. A zero window is proof of receiver pressure at TCP, not proof of which receiver subsystem caused it.

### Avoid snapshot diagnosis

Both windows change. A single `ss` line can catch slow start, idle restart, recovery, or a moment just after the application drained the socket. Sample through the transfer, preserve timestamps, and compare at least several RTTs.

For a 150 ms path, a 500 ms polling interval sees only a few RTTs per sample. That is enough for a coarse diagnosis, not a reconstruction of every transition. Use packet timestamps for sequence-level analysis and endpoint telemetry for internal state.

## A dated operational case: ESnet's measurement-host profile

![A dated host-tuning case maps path BDP to separate automatic and explicit buffer ceilings](/imgs/blogs/flow-control-vs-congestion-control-two-windows-one-pipe-6.webp)

The most useful public case is not an outage story. It is a scoped tuning profile from the people operating high-speed science networks.

### The case ledger

| Field | Evidence | Source |
| --- | --- | --- |
| Case | ESnet Fasterdata Linux test and measurement host tuning | [ESnet, January 15, 2025](https://fasterdata.es.net/host-tuning/linux/test-measurement-host-tuning/) |
| Event date | Guidance page dated January 15, 2025 | Page dateline |
| Source owner | Energy Sciences Network, Lawrence Berkeley National Laboratory | Page publisher |
| Mechanism | Large BDP requires receive and send buffer policy that permits a single TCP stream to keep enough bytes in flight | ESnet guidance and BDP derived here |
| Verified numbers | 120 MB buffer stated for 10 Gbit/s at 100 ms; 512 MiB core maxima and 256 MiB TCP autotuning maxima for 10 Gbit/s up to 200 ms | ESnet guidance |
| Transfer lesson | Tune to a named RTT, line rate, host role, and application behavior; do not present a measurement-host profile as a universal server default | Derived operational lesson |

ESnet states that a 10 Gbit/s path at 100 ms needs 120 MB of buffer. The ideal decimal BDP calculation gives 125 MB, so the published number is plainly a practical rounded guideline, not a claim of bit-exact capacity. Its configuration raises `net.core.rmem_max` and `wmem_max` to 512 MiB while setting the autotuning maxima in `tcp_rmem` and `tcp_wmem` to 256 MiB for 10 Gbit/s paths up to 200 ms.

The important detail is the workload scope. ESnet describes Linux test and measurement hosts running tools such as `iperf`, `iperf3`, or `nuttcp`, and assumes serialized single-stream tests. This is not the same risk model as an Internet-facing proxy with hundreds of thousands of connections. The profile buys room for large flight on long paths. Its blast-radius multiplier is per-socket growth under concurrency.

There is also a useful separation of concerns in the published settings. Core maxima allow large explicit `setsockopt` requests. TCP vector maxima permit autotuning to grow. This mirrors the two Linux paths described earlier. Copying only one line may fail depending on the benchmark application's socket behavior.

The transferable control is a review, not a magic constant:

1. Measure the target RTT distribution.
2. Name the required per-flow rate.
3. Calculate the BDP.
4. Inspect whether the application sets socket buffers.
5. Set ceilings with explicit memory-concurrency budgets.
6. Verify the wire advertisement and achieved flight on a new connection.

That process scales from a lab namespace to a dedicated data-transfer node. It also tells a service owner when not to copy the profile.

## Capacity planning the memory side

Window tuning is often presented as free throughput unlocked by one number. It is actually permission for TCP to retain more state per active flow. The permission is valuable, but it belongs in a host-level capacity model.

### Ceiling is not allocation

Setting `tcp_rmem[2]` to 256 MiB does not immediately allocate 256 MiB for every socket. Autotuning grows eligible buffers in response to observed need, subject to memory conditions. That distinction is why a high ceiling can be reasonable on a dedicated transfer host.

It is still unsafe to multiply nothing. A busy receiver with many simultaneous long-fat flows can have many sockets grow at once. Kernel accounting includes payload and metadata, while cgroups and the host compete for the same physical memory. Under pressure, TCP moderates allocations. Throughput can therefore degrade exactly when concurrency rises, even though the configured per-socket maximum did not change.

A first-pass explanatory capacity model is:

$$
M_{receive} \approx N_{active} \times B_{average}
$$

Here $N_{active}$ is the number of concurrently active receiving sockets and $B_{average}$ is their average allocated receive memory, not the configured maximum. This is an approximation for planning, not a Linux accounting equation. Measure `skmem` distributions and `/proc/net/sockstat` under representative traffic before turning it into a production limit.

Suppose 200 active transfer sockets average 64 MiB of receive allocation during peak traffic. Payload-side arithmetic alone is:

$$
200 \times 64\ \text{MiB} = 12{,}800\ \text{MiB} = 12.5\ \text{GiB}
$$

That is before send buffers, kernel structures, application heaps, page cache, and replicas of the service. The derived number does not say Linux will allocate exactly 12.5 GiB. It says a host with 8 GiB of memory cannot safely assume every one of those flows will sustain that average allocation.

### Size for the service objective, not the NIC label

A 10 Gbit/s NIC does not imply every flow needs a 187.5 MB receive window. If a service objective calls for 500 Mbit/s per flow at 150 ms, the idealized requirement is:

$$
W_{required} \approx \frac{500{,}000{,}000 \times 0.150}{8} = 9{,}375{,}000\ \text{bytes}
$$

That is about 8.94 MiB. A 32 MiB effective window leaves useful headroom without treating every connection as a full-line-rate bulk transfer. Conversely, a replication service expected to fill the whole circuit from one stream has a different requirement.

| Workload objective | RTT | Derived ideal flight per flow | 100 simultaneously active flows | Source |
| --- | ---: | ---: | ---: | --- |
| 100 Mbit/s per flow | 50 ms | 625,000 bytes | 62.5 MB payload-side arithmetic | Derived here: $W=T R/8$ |
| 500 Mbit/s per flow | 150 ms | 9,375,000 bytes | 937.5 MB payload-side arithmetic | Derived here: $W=T R/8$ |
| 2 Gbit/s per flow | 80 ms | 20,000,000 bytes | 2.0 GB payload-side arithmetic | Derived here: $W=T R/8$ |
| 10 Gbit/s per flow | 150 ms | 187,500,000 bytes | 18.75 GB payload-side arithmetic | Derived here: $W=T R/8$ |

The concurrency column is not a recommended memory allocation. It simply multiplies ideal flight by 100 to reveal the order of magnitude. Real allocation contains overhead, while not every socket reaches peak flight simultaneously.

### Separate connection populations when their jobs differ

One global TCP vector applies to diverse applications in the same network namespace unless applications set their own options. A dedicated transfer node can reasonably carry a different policy from a latency-sensitive API host. Containers with separate network namespaces can carry distinct namespaced settings, although the underlying host memory remains shared.

If only one application needs large fixed buffers, a socket-level configuration may narrow the blast radius better than a global default. If most long-lived flows benefit from adaptive growth, autotuning with a justified ceiling is simpler. Either way, inventory libraries and proxies that call `SO_RCVBUF`. An unnoticed fixed value can defeat the intended host policy.

### Observe distributions, not one socket

During a canary, collect socket memory and connection counts across the workload interval. Look for the median, high percentiles, and total. Correlate them with goodput and memory-pressure signals. A useful rollout asks three questions:

1. Did the target flows advertise and use more flight?
2. Did their completion time improve at the expected RTTs?
3. Did host or cgroup memory pressure worsen under realistic concurrency?

If the first answer is no, the configured knob may not own the socket. If the first is yes and the second is no, another limiter took over. If both are yes but the third also worsens materially, the throughput gain has an explicit capacity price. That is an engineering trade, not a tuning failure.

## Failure patterns that look alike from the application

An application often exposes only bytes per second and request duration. Several unrelated mechanisms collapse into the same two metrics. Senior diagnosis keeps competing explanations alive until a measurement separates them.

### Pattern 1: stable plateau with no retransmissions

A nearly horizontal goodput line, stable RTT, and negligible retransmissions are consistent with receive-window limitation, but do not prove it. The sending application might be producing at a fixed rate. A token-bucket shaper might be pacing the flow. Storage might deliver a stable 40 Mbit/s.

The discriminating observation is whether flight repeatedly reaches the peer's advertised right edge. A capture may mark `tcp.analysis.window_full`, and sender state should show a congestion allowance larger than the receiver allowance. If the sender never approaches either window, look upward toward the application and downward toward pacing.

### Pattern 2: zero-window bursts and sawtooth throughput

When the receiver application consumes in batches, the wire can alternate between data bursts and zero-window stalls. Average throughput may resemble a congested sawtooth, yet retransmission and queue evidence remain quiet. The capture shows `win 0`, probes, and reopening updates. Receiver profiling shows the read loop sleeping or working elsewhere.

A larger receive buffer can turn short stalls into continuous flight. It cannot make an indefinitely blocked consumer healthy. The durable fix may be separating network reads from slow processing, bounding downstream queues, or applying application-level backpressure before TCP memory is exhausted.

### Pattern 3: falling `cwnd` with open receiver space

If the peer advertises ample capacity while sender `cwnd` contracts around loss or ECN, the path-facing control loop owns the plateau. Raising receive memory adds no permission because `cwnd` remains the minimum. Inspect retransmissions, RTT, qdisc backlog, and path changes. Then choose an intervention appropriate to the loss and queue mechanism.

Avoid the reverse folklore too. A large `cwnd` does not mean the path is uncongested forever. It is sender state based on recent evidence. A route change, competing flow, or new queue can alter the safe flight during the same connection.

### Pattern 4: large windows with low flight

This is the application-limited case. Both TCP controllers permit more data, but the send queue is empty or the host cannot feed it. Common causes include synchronous file reads, serialization on a single CPU, lock contention, garbage collection, request generation that waits for application acknowledgments, or an intentionally paced producer.

The fix is above TCP. Raising either window changes unused permission. Check whether `ss` reports `app_limited`, whether the send queue stays near zero, and whether application or storage telemetry explains the cadence. For RPC workloads, the application protocol may allow only one small request outstanding, creating its own window far below TCP's.

### Pattern 5: enough average window, insufficient tail headroom

A receive buffer can be large enough for average RTT and target throughput yet still close during scheduling pauses or RTT excursions. Suppose a flow targets 2 Gbit/s at a median 80 ms RTT. Its ideal BDP is 20 MB. If RTT sometimes reaches 120 ms, the same rate requires 30 MB. A 24 MB effective ceiling is adequate at the median and limiting in the tail.

This is why production tuning uses an RTT distribution and a scheduling-jitter budget rather than one ping average. It also explains intermittent low-throughput reports: the same socket policy crosses from adequate to limiting when path delay changes.

| Pattern | Window evidence | Other evidence | First owner to investigate | Source |
| --- | --- | --- | --- | --- |
| Stable receive limit | Flight reaches peer `rwnd`; `cwnd` is larger | RTT and retransmissions stable | Receiver buffer policy and read loop | Reproducible with the lab below |
| Bursty zero window | `rwnd` reaches zero and reopens | Persist probes, receiver pauses | Receiver application scheduling | [RFC 9293 §3.8.6.1](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.8.6.1) |
| Congestion limit | `cwnd * mss` is smaller | Loss, ECN, RTT, or recovery state | Sender congestion loop and path | [Linux `ss(8)`](https://man7.org/linux/man-pages/man8/ss.8.html) |
| Application limit | Neither window is reached | Empty send queue or `app_limited` | Sending application or storage | [Linux `ss(8)`](https://man7.org/linux/man-pages/man8/ss.8.html) |
| Tail-only window limit | Window is enough at median RTT, small at tail RTT | Throughput drops with RTT excursion | Capacity planning and path variance | Derived from $W=T R/8$ |

## A diagnostic runbook for low TCP goodput

![Decision tree separating receiver-window, congestion-window, application, and RTT limits](/imgs/blogs/flow-control-vs-congestion-control-two-windows-one-pipe-7.webp)

**Rule of thumb: compare limits in bytes over the same interval.** “The window is 100” is meaningless until we know whether that means segments, raw header units, scaled bytes, or socket allocation.

### Step 1: prove the symptom belongs to transfer

Separate connection setup from body transfer. For HTTP, use `curl -w` fields such as `time_connect`, `time_appconnect`, `time_starttransfer`, and `time_total`. A large gap before first byte is not automatically a TCP flight-window problem. A large-body transfer that starts promptly and plateaus is a better candidate.

Confirm sender input and receiver output can keep up. A disk that reads at 40 Mbit/s creates the same application throughput as a 750,000-byte window over 150 ms, but bytes in flight and advertised windows will tell a different story.

### Step 2: calculate the required flight

Use the desired single-flow goodput, not the interface label:

$$
W_{required} \approx \frac{T_{target} \times R}{8}
$$

For a target of 2 Gbit/s at 80 ms:

$$
W_{required} \approx \frac{2{,}000{,}000{,}000 \times 0.080}{8} = 20{,}000{,}000\ \text{bytes}
$$

This is an explanatory minimum before overhead and variation. It gives the investigation a scale. A 256 KiB receive window is suspicious; a 64 MiB window probably is not the first constraint for that target.

### Step 3: decide which side owns the smaller bound

On the sender, compare `cwnd * mss` to the scaled peer advertisement visible in a capture. Also estimate bytes in flight from sequence progress or analyzer fields. Then branch:

- If peer `rwnd` is smaller and repeatedly reached, investigate receiver socket capacity and read cadence.
- If `cwnd` is smaller and loss or ECN evidence exists, investigate the path and congestion-control state.
- If neither is reached and `app_limited` appears, investigate the sending application, storage, and scheduling.
- If both look large enough, verify RTT, pacing, qdisc, CPU, offload interpretation, and the measurement tool.

Do not compare a packet's raw 16-bit window field directly with `cwnd * mss`. Apply the negotiated scale, or use an analyzer's calculated value after capturing the SYN exchange.

### Step 4: change one owner

For receiver limitation, change the receiver. For congestion limitation, change or repair the path or sender algorithm with a controlled experiment. For application limitation, make the producer supply bytes consistently. Changing all three destroys the diagnostic value of the test.

When applying a receive-buffer policy, open a new connection. Window scaling and initial socket behavior are established at connection setup. Confirm the application did not override the global policy.

### Step 5: verify the new failure boundary

A successful buffer increase should raise available receive flight and goodput until another limiter appears. That next limiter may be `cwnd`, line rate, CPU, memory bandwidth, storage, or the application. The expected result is not always 10 Gbit/s. The expected result is that the old 40 Mbit/s ceiling disappears and the evidence points to the next bound.

| Change | Desired observation | New risk | Source |
| --- | --- | --- | --- |
| Raise `tcp_rmem[2]` for autotuned sockets | Larger scaled advertised window and flight on a new connection | More memory growth per active socket | [Linux IP sysctl documentation](https://www.kernel.org/doc/html/latest/networking/ip-sysctl.html#tcp-rmem) |
| Raise `net.core.rmem_max` for explicit `SO_RCVBUF` | Application request is no longer capped at the old core maximum | A process can request larger socket allocations | [Linux `socket(7)`](https://man7.org/linux/man-pages/man7/socket.7.html) |
| Make the application read more consistently | Fewer zero-window intervals and lower unread queue | Downstream work may become the next bottleneck | Reproducible with the lab and application profiling |
| Change congestion control | Different `cwnd`, pacing, and recovery behavior when `cwnd` was limiting | Fairness and queue behavior change | [Congestion-control sibling](/blog/software-development/networking/congestion-control-cubic-bbr-and-what-changing-it-actually-does) |

## Run it yourself

### Question

With a 150 ms RTT and a 200 Mbit/s bottleneck, does increasing one explicit receive-window request from 750,000 bytes to 32 MiB remove the roughly 40 Mbit/s window ceiling without changing the path?

### Preconditions

Use Linux with `iproute2`, `iperf3`, `tcpdump`, `tshark`, and permission to create network namespaces and qdiscs. The commands assume the canonical `c` and `s` namespaces from [the series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), with `c0` at `10.77.0.1/30` and `s0` at `10.77.0.2/30`.

Namespace, qdisc, capture, and sysctl mutations require root or appropriate capabilities. Run them only in the disposable `netlab` namespaces, never on an unspecified production interface. Packet captures may contain sensitive payload; this lab restricts capture to one port and a short duration.

Preflight proves names, routes, tools, and current policy:

```bash
set -euo pipefail

SLUG=flow-control-vs-congestion-control-two-windows-one-pipe
OUT="netlab/out/$SLUG"
mkdir -p "$OUT"

sudo ip netns list | grep -E '^(c|s)( |$)'
sudo ip -n c link show c0
sudo ip -n s link show s0
sudo ip -n c route get 10.77.0.2
sudo ip -n s route get 10.77.0.1
command -v iperf3 tcpdump tshark

sudo ip netns exec s sysctl net.ipv4.tcp_moderate_rcvbuf
sudo ip netns exec s sysctl net.ipv4.tcp_rmem
sudo ip netns exec s sysctl net.core.rmem_max
```

Install one symmetric delay and rate model. Each endpoint contributes 75 ms one-way delay, producing about 150 ms RTT. The 200 Mbit/s rate keeps the experiment practical on a laptop while leaving the expected 40 Mbit/s baseline window-limited.

```bash
sudo ip -n c qdisc replace dev c0 root netem delay 75ms rate 200mbit limit 10000
sudo ip -n s qdisc replace dev s0 root netem delay 75ms rate 200mbit limit 10000

sudo ip netns exec c ping -c 8 -i 0.25 10.77.0.2
sudo ip -n c qdisc show dev c0
sudo ip -n s qdisc show dev s0
```

Read: inspect `ping`'s `rtt min/avg/max/mdev` line and both qdisc lines.

Expected: average RTT should usually be 148–160 ms in a Linux VM with light scheduler load. Both qdiscs should show `delay 75ms` and `rate 200Mbit`. If RTT is outside that range, use the measured RTT in the window-bound calculation rather than claiming the nominal value.

### Baseline

Start the server and a bounded capture in namespace `s`, then request a 750,000-byte socket window from the client. `iperf3 -R` makes the server send bulk data toward the client, so the client's receive setting controls the measured direction.

```bash
sudo ip netns exec s pkill -x iperf3 2>/dev/null || true
sudo ip netns exec s iperf3 -s -D -p 5201

sudo ip netns exec c timeout 25 tcpdump -i c0 -s 128 \
  -w "$OUT/baseline.pcap" 'host 10.77.0.2 and tcp port 5201' &
CAP_PID=$!

sudo ip netns exec c iperf3 -c 10.77.0.2 -p 5201 -R \
  -t 15 -O 3 -w 750000 --json > "$OUT/baseline.json"
wait "$CAP_PID" || true

jq '.end.sum_received.bits_per_second' "$OUT/baseline.json"
tshark -r "$OUT/baseline.pcap" -Y 'tcp.analysis.zero_window || tcp.analysis.window_full' \
  -T fields -e frame.time_relative -e tcp.window_size -e tcp.analysis.window_full
```

Read: in JSON, use `end.sum_received.bits_per_second`. In the capture, inspect calculated `tcp.window_size` and `tcp.analysis.window_full`. Keep the SYN packets in the capture so Wireshark knows the scale.

Expected: the receiver should usually report roughly 35–45 Mbit/s. The explanatory ceiling using exactly 750,000 bytes and a 150 ms RTT is 40 Mbit/s, but Linux accounting means `-w` is not a promise of an identical advertised payload window. Scheduler noise and actual RTT widen the range. Retransmissions should stay near zero on the lossless namespace path.

### Apply one change

Raise only the permitted explicit receive-buffer request, then use a 32 MiB request. The `net.core.rmem_max` value below follows Linux's doubled `SO_RCVBUF` accounting. Preserve the previous namespace-local value for reset.

```bash
OLD_RMEM_MAX=$(sudo ip netns exec c sysctl -n net.core.rmem_max)
printf '%s\n' "$OLD_RMEM_MAX" > "$OUT/old-rmem-max.txt"

sudo ip netns exec c sysctl -w net.core.rmem_max=67108864
sudo ip netns exec c sysctl net.core.rmem_max
```

This controlled treatment does not change RTT, rate, qdisc, congestion-control algorithm, payload direction, or test duration.

### Compare

Run a new connection with the larger request and collect the same fields:

```bash
sudo ip netns exec c timeout 25 tcpdump -i c0 -s 128 \
  -w "$OUT/treatment.pcap" 'host 10.77.0.2 and tcp port 5201' &
CAP_PID=$!

sudo ip netns exec c iperf3 -c 10.77.0.2 -p 5201 -R \
  -t 15 -O 3 -w 32M --json > "$OUT/treatment.json"
wait "$CAP_PID" || true

jq -n \
  --argjson baseline "$(jq '.end.sum_received.bits_per_second' "$OUT/baseline.json")" \
  --argjson treatment "$(jq '.end.sum_received.bits_per_second' "$OUT/treatment.json")" \
  '{baseline_bps:$baseline,treatment_bps:$treatment}'

tshark -r "$OUT/treatment.pcap" -Y 'tcp.analysis.zero_window || tcp.analysis.window_full' \
  -T fields -e frame.time_relative -e tcp.window_size -e tcp.analysis.window_full
sudo ip netns exec c ss -tinm dst 10.77.0.2
```

Read: compare `end.sum_received.bits_per_second`, the scaled advertised window, and any window-full events. During a live run, `ss -tinm` adds `cwnd`, `mss`, RTT, and socket memory evidence.

Expected: treatment should usually reach roughly 170–200 Mbit/s on a lightly loaded Linux VM, close to the configured 200 Mbit/s path rate, while baseline remains around 35–45 Mbit/s. The exact upper result varies with VM scheduling, qdisc behavior, CPU, and `iperf3` version. The path is unchanged; removing the small receive-window request exposes the next bottleneck, the 200 Mbit/s qdisc.

This lab demonstrates the window law at a manageable rate. It does not claim a laptop veth pair validates 10 Gbit/s hardware behavior. The 10 Gbit/s, 150 ms result is derived from the same units, while production validation must include NIC, CPU, offload, and memory measurements.

### Reset

Remove only this experiment's qdiscs, process, captures, and namespace-local sysctl change:

```bash
OLD_RMEM_MAX=$(cat "$OUT/old-rmem-max.txt")
sudo ip netns exec c sysctl -w "net.core.rmem_max=$OLD_RMEM_MAX"

sudo ip -n c qdisc del dev c0 root 2>/dev/null || true
sudo ip -n s qdisc del dev s0 root 2>/dev/null || true
sudo ip netns exec s pkill -x iperf3 2>/dev/null || true

rm -f "$OUT/baseline.pcap" "$OUT/treatment.pcap" \
  "$OUT/baseline.json" "$OUT/treatment.json" "$OUT/old-rmem-max.txt"
```

For a production host, use read-only commands first: `sysctl`, `ss -tinm`, `/proc/net/sockstat`, and a tightly governed capture. Change persistent sysctls through your configuration system only after calculating per-flow need and aggregate memory exposure.

## What to change, and what not to change

Reach for receive-buffer tuning when all of the following are true:

- A specific single-flow target and RTT imply a BDP above the effective receive flight.
- The receiver advertises the smaller active window.
- The sending application supplies data consistently.
- Loss and congestion evidence do not explain the plateau.
- Memory concurrency has been budgeted.
- A new connection proves the larger policy is negotiated and used.

Do not reach for it when:

- `cwnd` is the smaller limit and retransmissions or ECN identify path pressure.
- The sender is application-limited by disk, CPU, serialization, or write cadence.
- The transfer is shorter than the time needed for growth to matter.
- A middlebox or endpoint failed to negotiate window scaling.
- The receiver application repeatedly stops reading and a larger buffer would merely extend the stall.

Changing congestion control is appropriate when the path-facing limit is the problem and the workload has been defined. It is not a generic cure for low throughput. Increasing every buffer is appropriate only when the host role and concurrency justify the memory surface. Parallel streams are a workaround with fairness and operational costs, not proof that the network was congested.

Roll the change out like a capacity feature. Begin with one receiver population and preserve the previous sysctl values. Open fresh connections, since the Window Scale option is negotiated in the handshake. Compare completion time, scaled advertisements, flight, retransmissions, and total socket memory against an unchanged population. Hold offered load constant. If throughput rises but memory pressure or tail latency regresses, the result is not “TCP got faster.” The result is a quantified trade-off that needs a smaller ceiling, fewer concurrent bulk flows, or dedicated transfer hosts.

Document the direction explicitly in every careful performance review. Raising receive capacity on the service helps bytes traveling toward that service. It does not enlarge the remote receiver for responses sent in the opposite direction. Bidirectional replication can need policy review on both endpoints, but each direction should still be diagnosed independently. This one sentence in a runbook prevents a surprising number of changes on the wrong machine.

## Key takeaways

- TCP obeys both receiver flow control and sender congestion control. Effective flight is bounded by the smaller of `rwnd` and `cwnd`.
- `rwnd` protects receiver memory and application consumption. `cwnd` protects the network path. Similar throughput symptoms have different owners.
- A zero advertised window pauses new data. RFC 9293 persist probes preserve liveness until the receiver advertises space again.
- On Linux, receive autotuning grows no higher than `net.ipv4.tcp_rmem[2]`. Explicit `SO_RCVBUF` requests follow the `net.core.rmem_max` path and use doubled kernel accounting.
- A 10 Gbit/s path with 150 ms RTT has a decimal BDP of 187.5 MB. An effective 750,000-byte window yields an explanatory ceiling of exactly 40 Mbit/s.
- The fix for the stated autotuned case is to keep `tcp_moderate_rcvbuf=1` and raise `tcp_rmem[2]` above the derived need, such as 256 MiB for this path. If the application sets `SO_RCVBUF`, inspect and size `net.core.rmem_max` too.
- Verify with endpoint state and packet evidence. Compare `cwnd * mss`, scaled peer window, bytes in flight, RTT, retransmissions, and `app_limited` before touching a knob.

The permanent habit is to ask who is withholding permission. Once that answer is visible, TCP tuning stops being folklore and becomes accounting. The broader version of that discipline is the [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model).

## Further reading

- [RFC 9293: Transmission Control Protocol, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html)
- [RFC 7323: TCP Extensions for High Performance, September 2014](https://www.rfc-editor.org/rfc/rfc7323.html)
- [Linux kernel IP sysctl documentation](https://www.kernel.org/doc/html/latest/networking/ip-sysctl.html)
- [Linux `tcp(7)` manual](https://man7.org/linux/man-pages/man7/tcp.7.html)
- [ESnet Test/Measurement Host Tuning, January 15, 2025](https://fasterdata.es.net/host-tuning/linux/test-measurement-host-tuning/)
