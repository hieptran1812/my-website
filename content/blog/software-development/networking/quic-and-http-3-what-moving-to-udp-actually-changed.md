---
title: "QUIC and HTTP/3: What Moving to UDP Actually Changed"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Trace QUIC streams, migration, early data, and CPU cost from packet behavior to an HTTP/3 deployment decision."
tags:
  [
    "networking",
    "distributed-systems",
    "quic",
    "http-3",
    "udp",
    "congestion-control",
    "head-of-line-blocking",
    "connection-migration",
    "tls",
    "web-performance",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 42
image: "/imgs/blogs/quic-and-http-3-what-moving-to-udp-actually-changed-1.webp"
---

A mobile client starts loading a page over Wi-Fi, loses one packet, and then switches to cellular. The server responds quickly, but the client reports a slow page. Three distinct effects can hide behind that one complaint: a lost transport packet may hold up unrelated responses, a change of address may force a new connection, and a new connection may spend round trips establishing encryption before it can carry the request. QUIC changes all three, but it does not make the radio link reliable or make CPU work free.

The opening ladder below places the protocol choice inside a request. Read its round trips as *illustrative protocol steps*, not measured wall-clock durations. DNS, server work, and transfer remain on the path whichever HTTP version wins. The diagram is the mental model for this post: locate the time first, then decide whether QUIC owns it.

![A request latency ladder highlights transport and security setup while preserving DNS, server, and transfer context.](/imgs/blogs/quic-and-http-3-what-moving-to-udp-actually-changed-1.webp)

The path to this point started in [what happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The [HTTP/2 post](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer) explains how multiple requests share one TCP connection. QUIC changes the contract *below* those HTTP requests. It puts an encrypted, reliable, multiplexed transport in user space over UDP. HTTP/3 maps familiar HTTP semantics onto that transport. A GET remains a GET, but the units of loss recovery, address identity, and connection setup have changed. The authoritative specifications are [QUIC transport, RFC 9000 (May 2021)](https://www.rfc-editor.org/rfc/rfc9000.html), [QUIC with TLS, RFC 9001 (May 2021)](https://www.rfc-editor.org/rfc/rfc9001.html), [loss detection and congestion control, RFC 9002 (May 2021)](https://www.rfc-editor.org/rfc/rfc9002.html), and [HTTP/3, RFC 9114 (June 2022)](https://www.rfc-editor.org/rfc/rfc9114.html).

## 1. Put the improvement in the right layer

**A protocol name is not a latency measurement.** If a response waits behind a database query, swapping HTTP/2 for HTTP/3 does not shorten the query. If the browser is already using a warm TCP and TLS connection over a clean, low-latency network, there may be little connection-setup time left to save. If a page opens many simultaneous resources over a lossy path, removing one form of cross-stream blocking may matter. Those are different cases and deserve different measurements.

Here is a useful accounting model for a cold request. It is an explanatory approximation, not a timing equation stated by an RFC:

$$
T_{\text{request}} \approx T_{\text{resolution}} + T_{\text{establishment}} + T_{\text{server}} + T_{\text{delivery}}
$$

The terms name elapsed time for resolution, connection and security establishment, server processing, and response delivery. They can overlap in real clients, particularly when DNS, preconnect, and connection racing run concurrently. They are a checklist, not four stopwatch intervals that always add exactly. QUIC primarily changes establishment and delivery. A connection that survives an address change also avoids paying establishment a second time. The [latency-budget post](/blog/software-development/networking/the-latency-budget-speed-of-light-serialization-and-queueing) explains the propagation, serialization, and queueing terms still present underneath QUIC.

For a worked example, assume a path with a measured 80 ms round-trip time, no DNS delay, a cold HTTP/2 connection using a TCP handshake plus a TLS 1.3 handshake, and a cold HTTP/3 connection using a single QUIC handshake. A simplified serial setup model assigns two RTTs, 160 ms, to the former and one RTT, 80 ms, to the latter before protected application data. The *potential* difference is 80 ms. That is derived from the stated assumptions, not an observed browser result. TLS 1.3 can overlap work and clients may preconnect; cached connections erase the difference. A lost Initial or a Retry changes the QUIC timeline. We should measure a real page waterfall rather than multiplying an RTT and announcing an 80 ms production win. [RFC 9001's handshake mapping](https://www.rfc-editor.org/rfc/rfc9001.html#section-4) and [RFC 9114's connection-establishment discussion](https://www.rfc-editor.org/rfc/rfc9114.html#section-3.1) bound that model.

The distinction also matters for service operators. A CDN may speak HTTP/3 to the browser and HTTP/2 or HTTP/1.1 to the origin. Seeing `h3` at the client says nothing by itself about the origin hop. A reverse proxy may decrypt at the edge, open a new upstream connection, and add queueing or retries there. The [edge and re-encryption post](/blog/software-development/networking/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end) covers those boundaries. Before changing protocol configuration, annotate the path with the actual protocol on each hop.

| Observation | Candidate mechanism | Discriminating evidence |
| --- | --- | --- |
| Cold requests improve, warm ones do not | Saved establishment RTT | Split cold and warm timings; record negotiated protocol and connect/TLS timing |
| Several H2 responses pause together after loss | TCP byte-stream delivery | Capture TCP sequence gap and response progress on the same connection |
| One H3 response pauses while another advances | Independent QUIC stream delivery | QUIC transport trace or endpoint metrics with stream IDs and offsets |
| Both H3 responses slow at once | Shared path, congestion, or connection flow control | RTT, loss, congestion window, connection window, and receiver CPU |
| H3 cannot connect | UDP path or H3 endpoint unavailable | Explicit `--http3-only` attempt plus packet capture, then TCP comparison |

This table names tests, not promises. No single HTTP version wins all five rows. The HTTP API contract, payload design, and compression choices belong to [API performance](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency); this post stays with transport behavior.

## 2. UDP is the carrier, not the service contract

UDP gives an application datagrams addressed by IP and port. It does not give ordered bytes, retransmission, flow control, or congestion control. QUIC builds those properties above UDP and uses TLS 1.3 as part of its handshake and packet protection. HTTP/3 then maps request and response messages to QUIC streams. Calling HTTP/3 "HTTP over unreliable UDP" loses the most consequential part of the design: QUIC provides reliable, ordered delivery *within each stream* while permitting streams to advance independently. [RFC 9000, sections 2 and 12](https://www.rfc-editor.org/rfc/rfc9000.html#section-2) specifies streams and packets; [RFC 9114, section 3](https://www.rfc-editor.org/rfc/rfc9114.html#section-3) maps HTTP onto them.

At the wire, one UDP datagram can contain one or more QUIC packets. A QUIC packet has a packet number in a packet-number space, and its protected payload contains frames. A STREAM frame names a stream, an offset, and a slice of data. The stream offset controls delivery order for that stream. The packet number identifies transmission order and aids acknowledgment and loss detection. Those numbers solve different problems. A retransmission of stream bytes is sent in a *new* packet with a *new* packet number; the stream offset still identifies the data the receiver needs. [RFC 9002, section 2](https://www.rfc-editor.org/rfc/rfc9002.html#section-2) makes that separation explicit.

The transport is encrypted beyond the fields a middlebox needs to route datagrams. That reduces the amount of protocol state an intermediary can rewrite, a design response to [protocol ossification](/blog/software-development/networking/protocol-ossification-middleboxes-and-why-quic-looks-like-that). It also changes troubleshooting. `ss -ti` can expose TCP's congestion and retransmission state; it cannot decode a QUIC implementation's stream offsets or congestion window because those live in the process. A UDP socket in `ss -u` proves the process has a socket, not that the HTTP/3 request reached the application. Instrument the QUIC endpoint and export connection, path, packet-loss, flow-control, and stream-level signals. A packet capture still shows UDP reachability, packet sizes, addresses, and visible QUIC header fields, but most frames are protected. Endpoint qlog or equivalent transport tracing supplies the missing narrative.

There is also an operational reason for UDP. Deploying a new TCP transport feature depends on host kernels and often middlebox behavior. User-space QUIC libraries can ship with application or proxy releases. The cost is more work per packet in user space, possible extra copies, ACK scheduling, and a need to regain optimizations the mature TCP path already has. "User space" describes where much of the implementation lives, not a guarantee of slower operation. Kernel batching and offloads, implementation design, hardware, payload size, and traffic shape decide the actual cost. We will return to measurement rather than attach a universal multiplier.

### HTTP semantics stay recognizably HTTP

HTTP/3 retains methods, status codes, and field semantics. It changes framing and transport. The control stream carries settings; request streams carry request and response messages. QPACK compresses headers with a design that accounts for out-of-order QUIC stream delivery. That last detail matters because a header-compression scheme can create its own blocking even when transport streams are independent. [RFC 9114, section 4](https://www.rfc-editor.org/rfc/rfc9114.html#section-4) defines HTTP/3 streams and frames, while [RFC 9204 (June 2022)](https://www.rfc-editor.org/rfc/rfc9204.html) defines QPACK. We should not sell "no head-of-line blocking" as an unconditional property of every layer in the stack.

Nor does a QUIC stream mean a socket. Hundreds of requests may use streams on one QUIC connection, subject to peer-advertised stream counts, per-stream flow control, and connection-level flow control. A slow reader can still withhold credit. A congested path still limits the bytes the sender can place in flight. HTTP/3 removes the particular all-stream stall caused by TCP's one ordered byte stream. It does not remove shared capacity constraints.

## 3. Follow one lost packet through HTTP/2 and HTTP/3

**The question is whether the receiver can deliver already-arrived data from a different request.** In HTTP/2, request and response frames are multiplexed into one TCP byte stream. TCP exposes an ordered byte sequence to the HTTP/2 parser. If TCP segment carrying earlier bytes is lost, later bytes may arrive and sit in a receive buffer, but the parser cannot consume past the gap. A frame from an unrelated request, if located after that gap in the byte stream, waits as well. The application experiences transport head-of-line blocking even though HTTP/2 has independent stream IDs. This is the boundary described by the [HTTP/2 sibling](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer).

![A lost TCP byte range blocks later HTTP/2 frames, while QUIC permits another stream's complete data to be delivered.](/imgs/blogs/quic-and-http-3-what-moving-to-udp-actually-changed-2.webp)

In QUIC, the receiver tracks byte offsets per stream. Suppose packet P carries stream A bytes at offsets 0–999 and is lost. Packet Q carries stream B bytes at offsets 0–999 and arrives. The receiver can deliver B's data without waiting for A's missing bytes. The numeric offsets here are an illustrative example, derived only from the stated packet contents. They are not a specific capture or an MTU prescription. [RFC 9000, section 2.2](https://www.rfc-editor.org/rfc/rfc9000.html#section-2.2) says QUIC streams are ordered within each stream, and its [stream discussion](https://www.rfc-editor.org/rfc/rfc9000.html#section-2) notes that loss of a packet carrying frames from multiple streams can block *those* streams. A packet mixing A and B changes the example: if P held both, both wait for their respective missing offsets.

Consider a worked response pair. Resource A is a large image and resource B is a small stylesheet. We assume two streams, one lost transport packet containing only A data, a later packet containing all remaining B data, enough connection and stream receive credit, and a sender that already transmitted B. Under these assumptions B can become available to the HTTP/3 layer before A is repaired. Under HTTP/2, if B's frame bytes follow the lost TCP byte range on the single connection, B waits for TCP repair. This is a *delivery-order* result, not a claim that B finishes in a particular number of milliseconds. Loss detection and retransmission timing depend on acknowledgments, reordering, timers, and the path. [RFC 9002](https://www.rfc-editor.org/rfc/rfc9002.html) supplies the recovery algorithm.

<figure class="blog-anim">
<svg viewBox="0 0 860 300" role="img" aria-label="QUIC stream A waits for a lost packet while stream B continues to deliver; retransmission later completes A" style="width:100%;height:auto;max-width:860px">
<style>
.q3-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.q3-text{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}
.q3-small{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}
.q3-line{stroke:var(--border,#d1d5db);stroke-width:3}
.q3-b{fill:var(--accent,#6366f1)}
@keyframes q3-badvance{0%,18%{transform:translateX(0)}40%,100%{transform:translateX(450px)}}
@keyframes q3-arepair{0%,50%{opacity:0;transform:translateX(0)}55%{opacity:1;transform:translateX(0)}80%,100%{opacity:1;transform:translateX(450px)}}
@keyframes q3-aarrive{0%,74%{opacity:.35}82%,100%{opacity:1}}
.q3-bmove{animation:q3-badvance 10s ease-in-out infinite}
.q3-rmove{animation:q3-arepair 10s ease-in-out infinite}
.q3-adone{animation:q3-aarrive 10s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.q3-bmove,.q3-rmove{animation:none;transform:translateX(450px);opacity:1}.q3-adone{animation:none;opacity:1}}
</style>
<text class="q3-text" x="30" y="32">Independent QUIC stream delivery</text>
<text class="q3-small" x="30" y="57">One packet carrying stream A data is lost. Stream B still makes progress.</text>
<rect class="q3-bg" x="25" y="78" width="810" height="78" rx="10"/>
<rect class="q3-bg" x="25" y="174" width="810" height="78" rx="10"/>
<text class="q3-text" x="42" y="107">Stream A</text>
<text class="q3-text" x="42" y="203">Stream B</text>
<line class="q3-line" x1="245" y1="126" x2="735" y2="126"/>
<line class="q3-line" x1="245" y1="222" x2="735" y2="222"/>
<circle class="q3-b q3-rmove" cx="262" cy="126" r="13"/>
<text class="q3-small" x="320" y="145">loss waits for retransmission</text>
<circle class="q3-b q3-bmove" cx="262" cy="222" r="13"/>
<text class="q3-text q3-adone" x="744" y="130">delivered</text>
<text class="q3-text" x="744" y="226">delivered</text>
<text class="q3-small" x="30" y="284">Shared congestion and flow-control limits can still slow both streams.</text>
</svg>
<figcaption>Stream B can deliver while Stream A waits for its lost data; A completes after retransmission. Both remain subject to shared connection limits.</figcaption>
</figure>

The animation makes the condition visible. Watch the delivered data on B continue while A waits for retransmission. The end frame is still a legitimate state: both streams eventually complete, assuming recovery succeeds. It is tempting to draw B flying past every loss event, but that would be false when the lost packet contains B data, when B has not been sent, when connection-level flow control is exhausted, or when congestion control stops the sender. Independent *receive ordering* is narrower than independent network capacity.

QUIC's loss response can reduce the sending rate for the whole connection. Multiple streams share one congestion controller on a path and typically share the same bottleneck. If packet loss signals congestion, the sender may have less capacity for A *and* B even though B's arrived bytes are deliverable. This is why the right sentence is "QUIC removes transport-level head-of-line blocking across streams," followed by the conditions, rather than "QUIC makes streams independent." [RFC 9002, section 7](https://www.rfc-editor.org/rfc/rfc9002.html#section-7) describes the shared, per-path congestion controller. The [flow-control versus congestion-control post](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe) shows why a receiver window and a path window can produce similar looking stalls for different reasons.

### What a trace can and cannot prove

An ordinary packet capture can show one or more UDP datagrams missing from an observed sequence of visible packet numbers in circumstances where those packet numbers can be decoded. It cannot generally tell you which protected STREAM frames were inside the lost datagram. Even packet numbers can be obscured by header protection, and a capture at one host is not a full end-to-end account. A QUIC endpoint trace with stream IDs, offsets, ACKs, bytes in flight, and congestion state is a better instrument for this question. Compare stream B's delivered-offset timeline with stream A's missing-offset timeline. If both pause, inspect shared congestion, connection flow control, application scheduling, and whether the lost packet contained data from both. A single waterfall screenshot does not distinguish these mechanisms.

The common mistake is to benchmark one large file and call the result a head-of-line test. One large file uses one response stream. QUIC still needs to repair missing bytes within that stream. Its cross-stream advantage is tested with concurrent streams whose packets can be lost separately. A better experiment records completion or progress for at least two responses under controlled loss, not aggregate throughput of one transfer. The lab later in this post establishes the protocol and packet-path prerequisites; it deliberately does not pretend that a public endpoint and an uncontrolled internet path constitute a clean head-of-line benchmark.

## 4. Packet recovery and congestion did not disappear

QUIC acknowledges packets and retransmits information the peer still needs. It does not retransmit a packet with the same packet number. The sender can place lost STREAM data in a new packet, and the receiver attaches it to the original stream offset. This distinction is useful in traces: a packet-number gap can coexist with a stream that eventually has no byte gap. The packet number tracks transmission; the stream offset tracks delivery. [RFC 9002, section 2](https://www.rfc-editor.org/rfc/rfc9002.html#section-2) states this separation and explains why it removes ambiguity around retransmissions.

A sender declares loss using acknowledgment evidence or a time threshold, with tolerance for reordering. It also arms a probe timeout when acknowledgments fail to arrive. The exact threshold and timing are protocol variables, not something to infer from one screenshot. [RFC 9002, section 6](https://www.rfc-editor.org/rfc/rfc9002.html#section-6) specifies loss detection. As with TCP, unnecessary retransmission wastes capacity, while waiting too long hurts latency. QUIC's design gives the endpoint the signals and freedom to evolve its controller, but an operator still has to prove that a tuning change improves the intended workload.

Flow control and congestion control remain different. A stream window protects the receiving application from receiving more stream data than it can buffer. A connection window protects aggregate receive memory. The congestion window protects the network path by limiting bytes in flight. If the server runs out of connection-level credit because the client does not read, an unrelated stream may stall despite having separate offsets. If path congestion reduces the sending rate, all active streams compete for fewer bytes. The [two-windows post](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe) develops this distinction, and [RFC 9000, section 4](https://www.rfc-editor.org/rfc/rfc9000.html#section-4) specifies QUIC flow control.

A practical diagnostic should therefore collect four time series, aligned on one clock: stream progress, connection receive credit, bytes in flight with congestion window, and path RTT/loss. If stream B's offset advances through A's loss, independent delivery worked. If neither advances while receive credit is zero, investigate the reader. If neither advances while bytes in flight is at the congestion window and RTT rises, investigate the path and sender pacing. If the network is idle and credit is available, look at application scheduling or CPU. These are causal interpretations, not a fixed decision tree that replaces traces.

### A numerical guardrail for shared capacity

Suppose a path has a measured minimum RTT of 80 ms and an available bottleneck rate of 20 Mbit/s. As a rough bandwidth-delay product model, the in-flight data needed to keep that link busy is:

$$
B \approx R \times T = 20\,\text{Mbit/s} \times 0.080\,\text{s} = 1.6\,\text{Mbit} \approx 200\,\text{kB}.
$$

Here $R$ is the assumed bottleneck rate, $T$ is the assumed minimum RTT, and $B$ is the approximate data in flight. The arithmetic is derived here, not a QUIC-required window value. It omits protocol overhead, queueing, and receiver constraints. If a controller allows substantially less than that amount in flight during a long transfer, it can underfill the path. If it puts much more into a shallow bottleneck queue, delay or loss may rise. The same arithmetic applies to TCP. QUIC's per-stream ordering does not repeal a shared bandwidth-delay product. The [congestion-control post](/blog/software-development/networking/congestion-control-cubic-bbr-and-what-changing-it-actually-does) goes deeper on choosing a controller and measuring the result.

RFC 9002 provides a NewReno-like example controller and allows a sender to choose another conforming algorithm. Its congestion controller is per path. That does not mean every HTTP/3 deployment uses the RFC example. A CDN or browser implementation may use another algorithm. To compare protocols fairly, record the actual sender controller, pacing, initial window, version, packet size, loss model, and CPU budget. Otherwise a headline that says "QUIC beat TCP" may really compare two implementation policies, or the reverse. [RFC 9002, section 7](https://www.rfc-editor.org/rfc/rfc9002.html#section-7) is the standard's boundary for that claim.

## 5. Connection identity can outlive an IP address

TCP connections are identified by their endpoint addresses and ports. When a phone moves from Wi-Fi to cellular, its source address and often source port change. Without a higher-level continuity mechanism, an established TCP connection cannot simply keep using the old four-tuple. QUIC includes connection IDs so the peer can associate packets from a new network path with the existing logical connection. That makes migration and NAT rebinding possible after the handshake. It does not guarantee seamless service under every proxy, load balancer, path, or application policy. [RFC 9000, sections 5 and 9](https://www.rfc-editor.org/rfc/rfc9000.html#section-9) define connection IDs and migration.

![Connection IDs preserve QUIC's logical connection while the client address changes and the new path is validated.](/imgs/blogs/quic-and-http-3-what-moving-to-udp-actually-changed-4.webp)

Walk the path carefully. First, the client and server establish a connection and the handshake is confirmed. The server provides connection IDs the client can use for future packets. The phone changes network and sends from its new address using an appropriate connection ID. The peer associates the packet with the existing connection. The endpoint validates reachability on the new path, using PATH_CHALLENGE and PATH_RESPONSE where appropriate. It updates path state and does not assume that the old congestion window and RTT estimate are safe on the new network. [RFC 9000, sections 8.2, 9, and 9.4](https://www.rfc-editor.org/rfc/rfc9000.html#section-9.4) specifies those steps and restrictions.

There are several qualifications worth making before promising "seamless handoff." An endpoint must not initiate migration before the handshake is confirmed. The peer may send a `disable_active_migration` transport parameter. A zero-length connection ID constrains useful migration. A server's routing fabric has to deliver packets for a connection ID to the right backend or maintain state that can follow it. The version of QUIC in RFC 9000 does not allow arbitrary migration to a new server address; the preferred-address mechanism is a special negotiated case. A NAT rebinding can look like a migration even when the user did not change radio networks. The endpoint must consider amplification and path validation, because sending large amounts of data to an unvalidated address can be abused. These are not footnotes to a marketing claim. They are the implementation conditions that determine whether the call actually survives.

Here is the time tradeoff as a model. Let a network change take $T_{\text{path}}$ to become usable. Let a fresh TCP plus TLS setup take $T_{\text{new}}$ after that, and let a QUIC path validation and controller warm-up cost $T_{\text{quic}}$. A successful QUIC migration can avoid a new handshake when $T_{\text{quic}} \lt T_{\text{new}}$, but it still pays $T_{\text{path}}$ and may need to rebuild sending rate. This is explanatory arithmetic, not a protocol guarantee. If the new network is blocked for UDP or routes connection IDs to a backend with no state, the attempt can fail and the client may need another path or protocol. The precise user-visible effect depends on buffering, application retries, and request idempotency as well as transport.

Consider a live audio or large download session. A mid-request network change may be less disruptive with a surviving QUIC connection because stream state and cryptographic context remain. For a tiny request issued only after the switch, the marginal benefit may be small relative to server and application work. To test migration in a client, log the connection ID, old and new source tuple, handshake status, path validation result, stream continuity, and time until useful data resumes. The HTTP status alone does not tell you whether migration succeeded: a retry on a new connection can produce the same status after a longer interruption.

### A routing trap at the edge

Connection IDs are visible enough for routing, but that puts a requirement on the edge. If packets arrive on different ingress servers after a network change, either the load balancer must route the ID to the connection owner or the implementation must have a way to transfer or reconstruct state. The connection ID is not a magic cross-datacenter session store. It also has privacy implications: a static identifier can link activity across addresses. RFC 9000 recommends fresh IDs in migration-related contexts, subject to endpoint policy. This is a direct connection to [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7): request routing and transport connection routing are different decisions, and the proxy has to know which one it is making.

When debugging a migration complaint, start by distinguishing address movement from connection movement. A new source address with a continuous QUIC connection ID and stream state suggests migration. A new handshake and new stream IDs suggest reconnection. A frozen stream with successful path validation suggests the bottleneck has moved to congestion, flow control, or application work. Do not conclude that migration failed solely because throughput briefly fell after the switch. The new path may have different RTT, loss, MTU, and capacity, and RFC 9000 tells the sender to reset path estimates for precisely that reason.

## 6. Zero-RTT saves waiting, with a replay contract

The most misunderstood QUIC claim is "zero-latency connection." Zero-RTT means a client with suitable information from an earlier connection can send some application data before completing a new handshake. It does not make the network propagation delay zero, does not remove server processing, and does not apply automatically to a first visit. A normal new QUIC connection still establishes cryptographic keys and validates the server. [RFC 9001, section 4](https://www.rfc-editor.org/rfc/rfc9001.html#section-4) describes handshake levels; [section 5.6](https://www.rfc-editor.org/rfc/rfc9001.html#section-5.6) states the restrictions on 0-RTT application data.

![A fresh QUIC handshake and a resumed attempt differ in when request data may be sent and in replay risk.](/imgs/blogs/quic-and-http-3-what-moving-to-udp-actually-changed-5.webp)

In a first-contact flow, the client sends an Initial, receives handshake material, verifies the server, and obtains keys for protected application data. In a resumed flow, prior session state lets the client construct early data immediately. That early data can be rejected and resent after the handshake. It can also be replayed by an adversary in ways ordinary post-handshake data cannot. The server and application have to agree on what is safe to process early. RFC 9001 explicitly says an application must request use of 0-RTT for application data. [RFC 8470 (September 2018)](https://www.rfc-editor.org/rfc/rfc8470.html) defines HTTP early-data handling, including the `425 Too Early` response and safe-method restrictions.

The important service-design rule is semantic. A read-only request for a public document may be safe to retry, but a GET that charges a payment or increments a counter is not safe merely because its method name is GET. A POST that creates a payment should not be sent as replayable early data unless the application has a deliberate replay-safe design, such as a correctly scoped idempotency key with durable deduplication. An intermediary has to carry early-data information and respect the origin's decision. `425` asks the client to retry after the handshake. These are application behaviors, not features a transport engineer can infer from the presence of a session ticket.

The potential saved time can be calculated without claiming a global result. Assume an 80 ms RTT and a resumed client that is permitted to send a safe request in early data. Compared with waiting one RTT for handshake confirmation before sending that request, the client can start the request roughly 80 ms earlier in this simplified timeline. The response still traverses the path, and the server still performs its work. A rejected early-data attempt may remove the benefit or cost more because it must be retried. A warm existing connection needs no new handshake, so 0-RTT does not improve that case. This is a scenario calculation under stated assumptions, not an IETF benchmark.

There is a second limit: transport parameters and HTTP settings from the previous session constrain what the client can assume in early data. A resumed connection is not permission to use arbitrary new capabilities. The implementation must reconcile accepted early data with the new handshake. [RFC 9000, section 7.4.1](https://www.rfc-editor.org/rfc/rfc9000.html#section-7.4.1) explains remembered transport parameters; [RFC 9114, section 7.2.4.2](https://www.rfc-editor.org/rfc/rfc9114.html#section-7.2.4.2) covers settings interactions. Operators should test ticket lifetime, server rotations, cross-region resumption behavior, early-data acceptance, and `425` handling rather than assume every second visit takes the fast path.

### What to record in a real deployment

Separate first contact, resumption without early data, accepted early data, rejected early data, and connection reuse. Those are five materially different populations. Count each separately. Report request method and route class without leaking sensitive parameters. Record whether the server accepted early data, whether a retry occurred, and whether the request was replay-safe by policy. Then compare end-to-end latency distributions within comparable populations. A site that reports one "HTTP/3 median" across these flows hides the causal mechanism.

Security and performance also meet at the edge. TLS termination determines who can inspect an HTTP request and enforce early-data policy. If a CDN terminates QUIC, the origin may never see the QUIC handshake, but it may receive a request forwarded from early data. The intermediary must follow the HTTP early-data rules and make the risk visible. This belongs alongside the [mTLS and service identity discussion](/blog/software-development/networking/mtls-and-service-identity-at-scale), where identity at each hop is explicit rather than inferred from a user-facing URL.

## 7. HTTP/3 discovery and fallback are part of the deployment

An HTTPS URL does not encode a promise that HTTP/3 is available. A server can advertise an alternative service using `Alt-Svc`, or a client can use another supported discovery mechanism. The client then attempts QUIC to the advertised host and UDP port, negotiates the `h3` application protocol, and sends HTTP/3 requests if the connection succeeds. If UDP connectivity fails, RFC 9114 says clients should attempt TCP-based HTTP. This fallback is a feature of a robust client and a source of measurement confusion. A request that succeeds after an HTTP/3 attempt may have completed over HTTP/2. [RFC 9114, section 3.1](https://www.rfc-editor.org/rfc/rfc9114.html#section-3.1) describes discovery and fallback.

When enabling HTTP/3, check both protocol advertisements and the network path. UDP must be allowed in the relevant direction through firewalls, proxies, load balancers, and edge infrastructure. An enterprise proxy may allow TCP 443 while blocking UDP 443. A carrier or hotel network may do the same. A stateless UDP rule can still be too narrow if replies are filtered. An advertised H3 endpoint can be stale or unhealthy. A browser may cache the alternative service and race connection attempts. None of those failures says the HTTP application route itself is broken; a TCP fallback can mask them until latency rises.

`curl --http3` is intentionally not a proof that the transfer used HTTP/3. The curl documentation says it may try an earlier HTTP version in parallel or on a sufficiently quick failure. Use `curl --http3-only` for a strict reachability test, then inspect the reported HTTP version. This behavior is documented in [curl's HTTP/3 guide](https://github.com/curl/curl/blob/master/docs/HTTP3.md) and [curl's command-line manual](https://curl.se/docs/manpage.html). Do not compare timing from an `--http3` run with an `--http2` run before checking which protocol actually completed.

The same caution applies to dashboards. A product metric named `h3_enabled` may mean the endpoint advertises H3, not that clients negotiated it. Track at least attempted, successful, fallback, and failed HTTP/3 connections, then segment by client, network type, region, and software version. For latency, keep cold and reused connections apart. For reliability, distinguish handshake timeout, transport close, HTTP status, and application timeout. For migration, record new-path validation and actual stream continuity. A single "H3 traffic share" number cannot explain whether users benefit or whether expensive attempts fall back.

### Three routing boundaries to annotate

First is browser to edge. That is where HTTP/3 most often enters the service. Second is edge to origin. That hop may use HTTP/2 or HTTP/1.1, possibly over a separate TCP connection. Third is service to service. A mesh or sidecar may terminate and initiate another protocol again. The request may therefore have an H3 client hop and a TCP origin bottleneck at the same time. The [service-discovery post](/blog/software-development/networking/service-discovery-from-dns-to-registries-to-xds) explains why changing an endpoint address is also a control-plane event; QUIC migration does not make stale service discovery correct.

A useful deployment record has one row per hop with peer, transport, HTTP version, TLS terminator, idle timeout, maximum UDP payload or MTU policy, flow-control limits, and observability owner. If you cannot fill the row, you cannot assign a client-visible pause to that hop. For example, seeing a QUIC retransmission near the browser says something about the access path. It does not prove the origin is healthy. Seeing a slow origin span says something about server work. It does not prove the browser did not first waste time on a failed UDP attempt.

## 8. The CPU and throughput frontier

**The strongest case for HTTP/3 is conditional, and so is the strongest objection.** QUIC removes TCP's cross-stream delivery dependency and lets transport logic evolve with application releases. At high packet rates, its receive path can cost CPU and constrain throughput. Packet processing, encryption, ACK generation, buffering, scheduling, and copies can all contribute. The right question is whether your workload sits on the latency side or the CPU and throughput side of that frontier, with the current implementation and hardware.

![An illustrative frontier locates latency benefits and receiver CPU or throughput costs without claiming benchmark data.](/imgs/blogs/quic-and-http-3-what-moving-to-udp-actually-changed-6.webp)

A concrete caution comes from Xumiao Zhang and colleagues' ["QUIC is not Quick Enough over Fast Internet" (October 2023 preprint)](https://arxiv.org/abs/2310.09423). In their tested fast-network settings, the UDP, QUIC, HTTP/3 stack delivered data rates up to 45.2% below the compared TCP, TLS, HTTP/2 stack. The authors observed the gap across lightweight clients and several browsers and attributed an important part to receiver-side processing overhead, including packet handling and user-space ACKs. The number is an upper result from their study, not an estimate for every site or a claim that HTTP/3 always consumes 45.2% more CPU. Their paper is evidence that implementation cost can dominate under fast-path transfer conditions, particularly when bandwidth rises. It does not erase the latency benefit of independent streams on lossy paths.

The opposite direction also has public evidence, with important historical limits. Google's [SIGCOMM 2017 paper on its then-deployed QUIC](https://research.google.com/pubs/the-quic-transport-protocol-design-and-internet-scale-deployment/) reports average Search response latency reductions of 8.0% for desktop and 3.6% for mobile users, plus YouTube rebuffer-rate reductions of 18.0% desktop and 15.3% mobile in the populations and period studied. Those are measurements of Google's pre-standardization deployment, applications, clients, and networks. They are not a guarantee for a present-day IETF QUIC v1 and HTTP/3 deployment at another company. The value of the result is methodological: evaluate a real user population and separate application outcomes from microbenchmarks.

Cloudflare supplied another useful counterweight in its [April 14, 2020 HTTP/3 versus HTTP/2 comparison](https://blog.cloudflare.com/http-3-vs-http-2/). In its measured page-load setup at that time, HTTP/3 trailed HTTP/2 by roughly 1% to 4% on average in North America, with similar regional observations, while the team continued working on congestion tuning, prioritization, CPU capacity, and throughput. The page and implementation were specific. This is not a universal ranking of the protocols. It is a public example of an operator reporting an inconvenient result rather than turning a standards feature into an automatic page-load win.

| Published result | Workload and bound | Source |
| --- | --- | --- |
| 8.0% lower average Search response latency for desktop users; 3.6% for mobile | Google's then-deployed QUIC and client populations, reported in 2017 | [Google SIGCOMM 2017 paper](https://research.google.com/pubs/the-quic-transport-protocol-design-and-internet-scale-deployment/) |
| 18.0% lower YouTube rebuffering on desktop; 15.3% on mobile | Google's video client populations and period in the same 2017 study | [Google SIGCOMM 2017 paper](https://research.google.com/pubs/the-quic-transport-protocol-design-and-internet-scale-deployment/) |
| HTTP/3 page load about 1% to 4% slower on average in North America | Cloudflare's tested page, regions, and 2020 implementation | [Cloudflare, April 14, 2020](https://blog.cloudflare.com/http-3-vs-http-2/) |
| Up to 45.2% lower data rate for QUIC stack | Tested fast-network client and workload configurations; upper result, not global average | [Zhang et al., October 2023](https://arxiv.org/abs/2310.09423) |

This table is intentionally not a leaderboard. The outcomes differ: Search response latency, video rebuffering, page load, and bulk data rate. The dates and stacks differ. You cannot rank the rows by percentage and conclude one protocol is faster. Define the user outcome, then design a matched comparison around that outcome.

### Price the packet rate before blaming encryption

Suppose an application transfers 1 Gbit/s of payload in packets averaging 1,200 payload bytes. A rough packet-rate model is:

$$
N \approx \frac{1{,}000{,}000{,}000\ \text{bits/s}}{1{,}200\ \text{bytes/packet} \times 8\ \text{bits/byte}} \approx 104{,}167\ \text{packets/s}.
$$

$N$ is packets processed per second. This arithmetic is illustrative and omits IP, UDP, QUIC, and encryption overhead. Smaller payloads increase packet rate at the same payload throughput. Every per-packet cost is multiplied by that rate. If a user-space receive path spends an additional 2 microseconds of one CPU core per packet in a particular implementation, the derived load is about 0.208 core seconds per wall second at this rate. That 2-microsecond input is a hypothetical scenario, not a published QUIC measurement. The calculation shows why a tiny per-packet overhead can matter at high rates and why batching, offload, and packet sizing deserve profiling. It does not prove that encryption is the culprit; profiling must isolate the work.

Packet rate alone is not the whole frontier. A page with small critical assets and intermittent radio loss may benefit from per-stream delivery without approaching a CPU ceiling. A video transfer at a high rate can become CPU-bound on the receiver even if handshake time is irrelevant after the first second. A connection with many tiny streams may pay header, scheduler, and ACK costs that one bulk stream does not. Compare under matched payload, concurrency, client device, RTT, loss, rate cap, server load, and implementation versions. Record the observed protocol at the end, not just the intended one at the start.

When evaluating an edge rollout, hold the content and routing policy steady. Split by client and network, and run a staged experiment with both user metrics and infrastructure metrics. User metrics should include time to first byte, completed page or request time, failures, and application-specific outcomes. Infrastructure metrics should include CPU cycles per useful byte or request, packet rate, UDP drops, memory, and tail latency under peak load. A client improvement that consumes enough edge CPU to raise queueing for everyone else can be a net regression. A capacity improvement that worsens mobile user latency can also be a regression. The [SLO and error-budget post](/blog/software-development/system-design/reliability-slos-error-budgets-and-graceful-degradation) owns the operational policy; this article supplies the transport questions those metrics must answer.

### User-space agility is a deployment trade

RFC 9002 deliberately gives a sender generic congestion signals and permits different conforming algorithms. A user-space implementation can update recovery, pacing, and controller behavior without waiting for every client or server kernel to change. That is valuable when tuning can be rolled out, observed, and reverted. It is dangerous if a performance experiment is rolled out without isolating its effect on fairness, retransmissions, and CPU. Different library versions may behave differently even when both report HTTP/3. A deployment record should pin the QUIC library, TLS backend, congestion controller, and relevant batching/offload settings. A protocol version string alone does not identify the implementation that consumed your CPU.

The practical frontier diagram therefore has no fixed coordinates. Move the workload toward lossy concurrent requests and the cross-stream benefit may become visible. Move it toward clean, high-bandwidth bulk transfer and packet-processing cost may dominate. Improve batching or offload and the frontier moves again. Change to a mobile client with a slower CPU and the receiver side may become decisive sooner. The only defensible "HTTP/3 is faster" sentence names a workload, comparison, client, path, and outcome.

### A rollout checklist that can falsify the hypothesis

Begin with a narrow hypothesis, such as "on high-RTT mobile paths, cold requests for pages with concurrent critical assets finish sooner when the client actually negotiates HTTP/3." That sentence specifies the population, connection state, content shape, outcome, and protocol condition. It can be false. A vague goal such as "turn on QUIC for speed" cannot tell a team what to measure after rollout. Keep the control population on the same edge routing, cache state, compression policy, and application release as the treatment. If the control and treatment visit different PoPs or cache tiers, a transport comparison has become a routing comparison.

Then choose guardrails before looking at the result. The first is correctness: HTTP response codes, completed bodies, retries, early-data rejections, and duplicate side effects. The second is reliability: connection-establishment failures, fallback rate, path-validation failures, and timeout tails. The third is capacity: CPU, packet rate, UDP drops, socket buffer pressure, and useful bytes per core. The fourth is user experience: first byte, critical-resource completion, page completion, or playback interruptions, as appropriate to the product. A p50 improvement coupled with a p99 failure rate increase is not a straightforward win. Likewise, an edge CPU increase can create queueing that only appears under the evening peak.

Keep a low-risk escape route. Browsers and capable HTTP stacks can fall back to TCP-based HTTP when UDP cannot connect; your deployment should preserve that endpoint during the experiment. Roll back the advertisement or traffic split if a segment regresses. Do not make a private service's only listener UDP based on a public browser fallback assumption: your own client may not implement that fallback. For service-to-service use, verify the client library's retry policy, request idempotency, and deadline budget explicitly. QUIC's connection migration is attractive for mobile clients, but a server fleet must route connection IDs consistently before relying on it as an availability feature.

Finally, archive enough metadata to explain a result six months later: client and server software versions, QUIC library, congestion controller, TLS configuration, edge region, network class, content set, experiment period, and the actual protocol negotiated. An unversioned graph can show a real regression without explaining which code path produced it. This is especially important for a user-space transport that can change behavior with an application release. The payoff is not bureaucratic completeness. It is the ability to distinguish a transport improvement from a CDN cache shift, a new client CPU bottleneck, or a change in how the browser schedules assets.

## 9. Two public deployments and one question they force us to ask

### Google, 2017: measure the user outcome

The organization was Google, and the event was its internet-scale QUIC deployment reported at SIGCOMM in August 2017. The direct source is the authors' [paper, "The QUIC Transport Protocol: Design and Internet-Scale Deployment"](https://research.google.com/pubs/the-quic-transport-protocol-design-and-internet-scale-deployment/). The authors measured outcomes for Google Search and YouTube clients using Google's deployed QUIC implementation of that period. The reported Search response-latency improvements were 8.0% for desktop and 3.6% for mobile on average. The reported reduction in YouTube rebuffer rate was 18.0% for desktop and 15.3% for mobile. Those are the source's numbers and populations, not values reconstructed from a figure or transferred to HTTP/3 in 2026.

The mechanism relevant here is a combination of reduced establishment waiting and avoiding TCP's cross-stream delivery stalls, embedded in a larger production transport design. We should not attribute every point of the observed gain to one mechanism unless the paper's experiment isolates it. The user-visible symptom was latency or playback interruption, not a transport counter. The trigger was an intentional protocol rollout, rather than an outage. The contributing conditions included Google's client base, server fleet, network paths, and implementation choices. The blast-radius concern was therefore rollout safety and heterogeneous path behavior: a change that improves one client population can regress another. The transferable guardrail is to instrument user outcomes and implementation costs together, with per-network segmentation. A packet-level theory tells us *where* to look. A controlled production experiment tells us whether the change helped users.

Notice what this case does not establish. It predates IETF QUIC v1 and the HTTP/3 RFC. It does not say a current library on a particular CPU will beat a current TCP stack. It does not resolve whether your edge, origin, or mobile client is the limiting host. That is why the paper is a starting point for measurement design, not a target percentage for a project plan. If a proposal promises an 8% gain by copying the headline, ask for the matched population, baseline protocol, and bottleneck attribution.

### Cloudflare, 2020: report the regression and its shape

The organization was Cloudflare. Its [April 14, 2020 engineering comparison](https://blog.cloudflare.com/http-3-vs-http-2/) tested HTTP/3 against HTTP/2 on a real page and reported that HTTP/3 was about 1% to 4% slower on average in North America in that setup, with similar results in several other regions. The source owner is the operator of the tested site and edge implementation. The user-visible metric was page-load time. The trigger was a performance comparison during HTTP/3 development, not a production incident. The report describes ongoing work in congestion tuning, prioritization, CPU capacity, and raw throughput. We should not turn that list into a single proven root cause for every observation: the post does not prove that the whole gap came from one knob.

The relevant mechanism is the frontier between protocol features and implementation cost. A theoretical saving from stream independence need not dominate on a particular page or path. Page-load time also depends on asset scheduling, critical resources, and browser behavior. Cloudflare's result is especially useful because it rebuts the shortcut "newer protocol equals lower latency" without requiring us to claim HTTP/2 is globally better. The blast-radius multiplier in a large edge rollout is broad client diversity: the average can hide a high-latency network win and a fast-network loss. The transferable control is a versioned A/B comparison with client, region, and path segments, plus both page outcomes and edge capacity. If the experiment shows a regression, keep the fallback and tune the implementation before making the protocol mandatory.

### The later CPU paper as a diagnostic clue

The [Zhang et al. October 2023 study](https://arxiv.org/abs/2310.09423) is not a Cloudflare or Google incident and should not be narrated as one. It is a measurement paper. Its up-to-45.2% data-rate gap under tested fast-network conditions gives us a hypothesis when bulk HTTP/3 transfers lag: profile receiver packet processing and user-space acknowledgment work. It does not identify the cause of Cloudflare's 2020 page-load result, and the two numbers should never be added or averaged. Together the three sources define a healthy engineering posture: protocol mechanisms explain possible improvements, real user experiments reveal the outcome, and profiling identifies an implementation bottleneck when the outcome contradicts the simple model.

## 10. Choose a diagnostic branch before changing a setting

When an H3 rollout disappoints, start with negotiated protocol, then ask whether the symptom is a connect failure, a per-stream stall, a connection-wide stall, or a CPU ceiling. Each branch points to a different measurement. The tree below is a triage order, not a proof by itself. A request can travel through several branches during its lifetime.

![An HTTP/3 diagnostic tree separates fallback, per-stream loss, shared congestion, and receiver CPU limits.](/imgs/blogs/quic-and-http-3-what-moving-to-udp-actually-changed-7.webp)

**No H3 connection.** Check server advertisement and the strict client attempt. If `--http3-only` fails but `--http2` succeeds, test UDP reachability to the advertised endpoint and inspect edge logs. Do not modify a production firewall based only on one laptop: a local proxy, client build, or network policy may explain the result. Keep a TCP fallback while investigating. If both fail, the problem is probably broader than HTTP/3.

**H3 connected, one response stalls.** Compare that stream's delivered offsets with other streams. Look for a missing offset and subsequent recovery. If other streams advance, the independent-delivery property is working even though the user still waits for the missing bytes of this response. If several streams share one lost datagram, more than one can stall. If only one request exists, there is no cross-stream benefit to observe.

**H3 connected, all responses stall.** Inspect connection flow-control credit, path loss and RTT, bytes in flight, sender pacing, and application dispatch. A shared congestion event or a client that stops reading can affect every stream. Avoid mislabeling the stall as "QUIC HOL" merely because several requests waited together. In HTTP/3, head-of-line blocking can still arise above transport through dependencies such as header processing or application scheduling; the exact layer matters.

**H3 connected, bulk throughput is low.** Profile both endpoints. Record packet rate, payload size, cycles per useful byte, UDP receive drops, socket buffers, and ACK work. Compare against a matched HTTP/2 test with the same content, clients, and path. If CPU saturates while network capacity remains unused, adding congestion-window capacity may only increase work or buffering. If CPU is idle and loss or RTT rises, examine the path instead. The [bufferbloat post](/blog/software-development/networking/bufferbloat-queueing-and-the-latency-you-added-yourself) explains why a growing queue can make a transfer look slow before loss counters rise.

**A mobile handoff fails.** Check whether the original handshake had been confirmed, whether active migration was disabled, whether a usable connection ID was available, and whether path validation completed. Then inspect edge routing for that ID and whether the new path can carry UDP. If the connection survived but speed fell, compare new-path RTT and congestion state before concluding migration was broken. The user needs a continuous application outcome; a successful PATH_RESPONSE alone is insufficient.

**Early data looks fast but creates duplicate actions.** Stop sending that route in 0-RTT and audit the application contract. Measure resumption and accepted early data separately from ordinary connection reuse. Respect `425 Too Early` and retry after the handshake when requested. A replay bug is an application correctness problem created by using an optional latency feature outside its allowed semantics.

## Run it yourself

### Question

Can a client prove that its request completed over HTTP/3 rather than silently falling back to HTTP/2, and can we identify a path where HTTP/3 is unavailable while TCP HTTPS still works? This lab tests protocol negotiation and fallback, one prerequisite for every stronger QUIC performance claim. It does not simulate cross-stream head-of-line behavior, migration, or 0-RTT.

### Preconditions

Use a Linux shell with `curl` built with both HTTP/2 and HTTP/3 support, plus `tcpdump` only if you want the optional packet check. A macOS reader can run the commands in the Linux environment introduced by [the first networking lab](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). Check `curl -V` and require `HTTP2` and `HTTP3` in its `Features:` line. curl's [HTTP/3 build guide](https://github.com/curl/curl/blob/master/docs/HTTP3.md) explains how to obtain a suitable build. The stock curl on the author's current macOS host reports version 8.7.1 and `nghttp2` but no HTTP/3 feature, so no result from that host is presented as if it were an H3 run. Use a known H3-enabled HTTPS endpoint you administer, or an H3-enabled public endpoint after verifying its current advertisement. The URL below uses `https://cloudflare.com/cdn-cgi/trace` as a concrete public target; availability and response contents can change. This is a read-only GET, and it does not require root. Packet capture needs permission and can collect sensitive traffic, so scope it to this target and omit it if unnecessary.

Set the target and print the effective client capability:

```bash
set -euo pipefail
LAB_URL='https://cloudflare.com/cdn-cgi/trace'
curl -V
curl -V | grep -E 'Features:.*HTTP2' >/dev/null
curl -V | grep -E 'Features:.*HTTP3' >/dev/null
```

If the second check fails, install an HTTP/3-capable curl and repeat preflight. Do not treat an unsupported-option error as a network result. This lab is about a client and remote endpoint, so it does not mutate the series' `netlab` namespace, interfaces, routes, or qdiscs. The canonical local topology from post #1 remains untouched.

### Baseline

Force HTTP/2 and record the negotiated version, status, remote address, TLS handshake completion time, first-byte time, and total time. Repeat three times to see ordinary network variance:

```bash
for run in 1 2 3; do
  curl --http2 --silent --show-error --output /dev/null \
    --connect-timeout 5 --max-time 20 \
    --write-out 'run=%{num_connects} http=%{http_version} code=%{response_code} remote=%{remote_ip} appconnect=%{time_appconnect} starttransfer=%{time_starttransfer} total=%{time_total}\n' \
    "$LAB_URL"
done
```

Read `http`, which should be `2` if the endpoint supports HTTP/2 and the client reached it. Read `code` to ensure an HTTP response completed. The timing fields are seconds and should be nonnegative with `total` at least as large as `starttransfer`; there is no honest universal millisecond range for a public endpoint. A reasonable *state* expectation is `http=2`, a successful HTTP status such as `200` if the endpoint is healthy, and three timings that vary rather than match exactly. `run=%{num_connects}` reports connection count within that single curl invocation; it is not the loop number. For a strict timing comparison, avoid treating these three independent processes as reused connections.

On September 30, 2026, one baseline invocation on the author's macOS host returned `http=2`, `code=200`, `time_appconnect=0.105481`, `time_starttransfer=0.157088`, and `time_total=0.157208` seconds. This is a reproducible one-run observation for that host and path, not a comparison with HTTP/3. The host curl build lacks HTTP/3, so the treatment requires the preflight's H3-capable Linux client.

### Apply one change

Change only the requested client protocol to strict HTTP/3. Keep URL, network, request method, timeout, and output fields the same. `--http3-only` matters: [curl documents](https://curl.se/docs/manpage.html) that `--http3` may fall back to an older version, while `--http3-only` does not.

```bash
for run in 1 2 3; do
  curl --http3-only --silent --show-error --output /dev/null \
    --connect-timeout 5 --max-time 20 \
    --write-out 'run=%{num_connects} http=%{http_version} code=%{response_code} remote=%{remote_ip} appconnect=%{time_appconnect} starttransfer=%{time_starttransfer} total=%{time_total}\n' \
    "$LAB_URL"
done
```

Read `http=3` and a completed HTTP status as the positive result. If it times out or reports a QUIC connection error, do not infer that the server application is down: compare the still-working HTTP/2 baseline, the current `Alt-Svc` response, and the UDP path. Strict H3 failure can mean blocked UDP, a stale or missing H3 service, a client-library mismatch, or transient endpoint trouble. The expected qualitative range is either completed `http=3` on an H3-capable path or a clear failure with no fallback; `http=2` must not be accepted as a successful strict-H3 treatment. Times can move in either direction across three public-internet samples and are not evidence of a performance ranking.

### Compare

Repeat the baseline, and inspect current advertisement. Keep this read-only:

```bash
curl --http2 --silent --show-error --head "$LAB_URL" | grep -i '^alt-svc:' || true
curl --http2 --silent --show-error --output /dev/null \
  --write-out 'http=%{http_version} code=%{response_code} total=%{time_total}\n' \
  "$LAB_URL"
```

Read the `Alt-Svc` line for an `h3` token and advertised UDP port. Its absence does not alone prove that an explicit HTTP/3 attempt must fail; other discovery or direct configuration can apply. Its presence does not prove UDP connectivity. Read the final `http=2` to confirm the baseline still works. For an optional packet check on a host where you have capture permission, first resolve the target and choose the actual remote address reported above, then capture only UDP traffic to that address and port during one strict-H3 call. For example, after substituting the observed IPv4 address for `REMOTE_V4`:

```bash
sudo timeout 10 tcpdump -i any -nn -s 128 \
  'udp and host REMOTE_V4 and port 443'
```

This optional capture proves only whether UDP datagrams leave and replies appear at that host. It does not decrypt streams or establish that an HTTP request succeeded. The `any` interface and `timeout` syntax assume Linux. Ask your network team before capturing on a managed host; even short captures can contain sensitive metadata. Do not run broad capture filters in production.

### Reset

The experiment changed one command-line option and made no persistent system mutation. Reset by dropping `--http3-only` and rerunning the baseline command. No namespace, firewall, route, or qdisc deletion is appropriate. If you started the optional `tcpdump`, stop that process with Ctrl-C. A failed treatment followed by a working baseline is useful evidence of H3 path or endpoint trouble, but it is only the first diagnostic branch. For production, prefer read-only edge logs, protocol-negotiation metrics, and a narrowly scoped client probe over changing firewall rules to make a demo pass.

## Key takeaways

- QUIC uses UDP as a carrier for encrypted, reliable streams. HTTP/3 uses those streams for HTTP messages; it is not an unreliable replacement for HTTP/2.
- Loss of data for one QUIC stream need not delay delivery of complete data from another. Shared packets, connection flow control, congestion, and application dependencies still matter.
- Connection IDs let an established connection survive some address changes, subject to handshake confirmation, path validation, routing, and policy. A new path still needs fresh congestion assumptions.
- Zero-RTT means eligible resumed requests can be sent early, with replay risk. It is not a first-contact shortcut or a blanket license for state-changing operations.
- The performance result depends on workload and implementation. Published user-latency wins coexist with published page-load regressions and high-speed receiver CPU limits.
- Measure negotiated protocol, end-user outcome, per-stream progress, path state, and CPU before declaring HTTP/3 a win or a failure.

## Further reading

- [RFC 9000: QUIC transport](https://www.rfc-editor.org/rfc/rfc9000.html), [RFC 9001: TLS in QUIC](https://www.rfc-editor.org/rfc/rfc9001.html), and [RFC 9002: recovery and congestion](https://www.rfc-editor.org/rfc/rfc9002.html), all published May 2021.
- [RFC 9114: HTTP/3](https://www.rfc-editor.org/rfc/rfc9114.html), published June 2022; [RFC 8470: HTTP early data](https://www.rfc-editor.org/rfc/rfc8470.html), published September 2018.
- [Google's 2017 deployment study](https://research.google.com/pubs/the-quic-transport-protocol-design-and-internet-scale-deployment/), [Cloudflare's 2020 comparison](https://blog.cloudflare.com/http-3-vs-http-2/), and [Zhang et al.'s 2023 fast-network study](https://arxiv.org/abs/2310.09423) show why every percentage needs a workload and date.
- [The senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) connects transport evidence to an end-to-end incident workflow.
