---
title: "HTTP/2 multiplexing, flow control, and the head of line that moved down a layer"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Trace HTTP/2 streams and windows on the wire, then distinguish a blocked receiver from a lost TCP segment."
tags:
  ["networking", "distributed-systems", "http-2", "multiplexing", "flow-control", "head-of-line", "hpack", "grpc", "tcp", "performance"]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer-1.webp"
---

A page has six independent resources. One large response slows down; the other five should keep making progress. The browser reports HTTP/2, so the team assumes the old head of line problem has gone away. A packet capture tells a more interesting story: several streams are active, yet every response pauses after one missing TCP sequence range. The request handler is healthy. The queue in the application is empty. The pause lives beneath the HTTP/2 scheduler.

![The latency ladder with HTTP/2 request and transfer work highlighted](/imgs/blogs/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer-1.webp)

The diagram above is the mental model: HTTP/2 changes how request and response bytes share a connection, but it does not replace DNS, TCP, TLS, or the server. A protocol label in a trace tells us which rules govern the exchange. It does not tell us which part consumed the time. We will descend from an HTTP request to frames, two receive windows, a single TCP byte stream, and the measurements that distinguish each stall. For the complete path before this point, start with [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The companion [HTTP/1.1 keep-alive post](/blog/software-development/networking/http-1-1-keep-alive-and-the-six-connection-tax) explains the connection behavior HTTP/2 set out to improve.

This post asks a narrow operational question: when six logical requests share one connection and progress stops, what exactly stopped? It could be the application, the stream receive window, the connection receive window, a proxy's stream limit, TCP loss recovery, or an overloaded endpoint. Each has a different safe response. Increasing a timeout because the call eventually completes hides the evidence. Increasing a window because throughput is low may only allow more memory to accumulate. The useful move is to identify the smallest boundary whose state changed before the pause.

## Where HTTP/2 lives in the latency budget

The canonical ladder is `DNS | TCP | TLS | request | server think | first byte | transfer`. It is an ordering of work, not a claim that all components run serially in every deployment. A warm connection may pay neither a DNS query nor a new handshake for a particular request. TLS may be resumed. A proxy can add another connection and another queue. Still, the ladder disciplines a performance conversation. An HTTP/2 flow-control stall belongs in response transfer or request-body upload. A TCP retransmission can affect first byte or transfer depending on which segment was lost. A slow handler belongs in server think, even if every request arrived over HTTP/2.

A convenient client-side timing set is `time_namelookup`, `time_connect`, `time_appconnect`, `time_starttransfer`, and `time_total` from `curl -w`. Their differences approximate parts of the ladder, with important caveats: reused connections yield timings that do not represent new handshakes, redirects add phases, and `time_starttransfer` includes server work plus network return time. The numbers are observations at the client, not packet-level proof of a flow-control state. Capture the negotiated protocol using `%{http_version}` too. If it says `2`, that establishes the protocol in this connection; it does not establish that multiplexing was effective for this workload.

The transfer part itself needs another decomposition. A response can wait to be scheduled by the HTTP/2 implementation, wait for its stream window, wait for the connection window, wait for TCP congestion or receive capacity, wait for retransmission, or wait for the sender application to produce more bytes. These are not interchangeable. The [flow-control versus congestion-control post](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe) gives the lower-level distinction. HTTP/2 adds receiver-advertised application windows above TCP's window and congestion machinery.

For a large gRPC response, this placement matters more than the marketing shorthand “multiplexed.” Suppose the handler writes messages rapidly but the client reads slowly. An HTTP/2 receiver can withhold `WINDOW_UPDATE`, and the sender must stop sending DATA when a window reaches zero. The service might record short handler time if it hands bytes to a buffer before the transport stalls, while the caller observes a long stream duration. Conversely, if the client consumes promptly but TCP retransmission pauses delivery, the HTTP/2 windows can remain open. Both look like low goodput from far away. Their wire signatures differ.

A useful first pass is to record the client timing, HTTP protocol, total response bytes, and TCP retransmission count over the same interval. Then inspect HTTP/2 frames if the transport and application disagree. Do not infer a window stall merely from a flat throughput graph. The receiver's advertised application credit and the sender's use of that credit are the decisive evidence.

## Frames, streams, and one TCP connection

![Interleaved HTTP/2 frames from independent streams entering one TCP byte stream](/imgs/blogs/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer-2.webp)

HTTP/2 calls a bidirectional sequence of frames a *stream*. A request and its response normally occupy one stream. A frame is the wire unit carrying control or part of the message: `HEADERS` for a header block, `DATA` for content, `SETTINGS` for connection settings, `WINDOW_UPDATE` for flow credit, `RST_STREAM` for a stream cancellation, and `GOAWAY` for connection wind-down. The names and semantics are specified in [RFC 9113, June 2022](https://www.rfc-editor.org/rfc/rfc9113). The frame header includes a stream identifier. An endpoint can therefore send part of stream 1, then part of stream 3, then return to stream 1 without requiring either message to finish first.

That property removes the HTTP/1.1 response ordering constraint on one connection. With HTTP/1.1 pipelining, response order followed request order; a long first response could hold later responses at the application protocol layer. HTTP/2 uses explicit frame boundaries and stream identifiers, so the receiver can reconstruct several messages from an interleaved frame sequence. Its `HEADERS` and `DATA` framing also means the application no longer has to guess message boundaries from connection closure. This is a significant improvement, especially when many small resources are ready at different times.

The TCP socket is nevertheless one ordered byte stream. The HTTP/2 frame parser sees bytes after TCP has put them in sequence. Imagine the sender writes a `DATA` frame for stream 1 and then a `DATA` frame for stream 3. If a TCP segment containing the end of the first frame is missing, TCP cannot hand the later bytes to the HTTP/2 parser as an independently delivered stream 3 frame. It may have received them into its out-of-order buffer, but the application reads an ordered stream. HTTP/2 can schedule frames independently only after TCP has delivered their bytes. The diagram's funnel from several logical lanes into one transport lane is the constraint.

The distinction between *concurrency* and *parallel delivery* is easy to blur. Several HTTP/2 streams can be open concurrently, and frames can be interleaved. The sender still has finite connection bandwidth, finite socket buffers, one congestion controller for this TCP connection, and one ordered delivery boundary. A second stream is not a second TCP connection. This is why an L4 load balancer that balances connections can pin many gRPC calls to one backend. The higher-level impact and design choices belong in [gRPC and Protocol Buffers for API design](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming); the wire reason is that the balancing unit at L4 is the connection, while the application unit is the stream.

Stream IDs also matter in captures. Client-initiated HTTP/2 streams use odd-numbered identifiers; the sequence can show whether several requests were genuinely in flight. An idle network trace with only one active stream does not test multiplexing. A verbose frame log should show multiple `HEADERS` frames with different IDs before all corresponding responses finish. The exact ordering of DATA frames is implementation and workload dependent; the RFC provides a framing contract, not a guarantee of fair scheduling. If a large response monopolizes the sender's output, ask whether the scheduler is willing and able to interleave, whether competing streams have queued data, and whether the connection window admits more DATA.

This is also why request cancellation is a protocol action rather than an automatic deletion of work. `RST_STREAM` says the stream no longer needs to continue on that HTTP/2 connection. It does not prove that a handler already dispatched to an upstream database has stopped. The application and proxy have to propagate cancellation and release resources. That difference became central in the public Rapid Reset case later in this post.

## Two receive windows, one hidden ceiling

![Each DATA byte consumes both its stream window and the connection window](/imgs/blogs/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer-3.webp)

**Rule of thumb: a stream may send DATA only while both its own window and its connection's window have room.** The receive window is a credit announced by the receiver. It bounds how much DATA the peer may send before the receiver grants more credit. HTTP/2 maintains one flow-control window for each stream and another for the connection. Both apply to DATA payloads. Control frames such as `WINDOW_UPDATE` need to get through even when DATA has exhausted the credit. [RFC 9113 section 5.2 and section 6.9](https://www.rfc-editor.org/rfc/rfc9113#section-5.2) define this mechanism. Flow control is hop by hop between the endpoints of an HTTP/2 connection; a proxy terminating one connection and opening another has separate window state on each side.

The starting value is exact, not an arbitrary benchmark. [RFC 9113 section 6.9.2](https://www.rfc-editor.org/rfc/rfc9113#section-6.9.2) says new streams begin with an initial flow-control window of 65,535 octets, and the connection window starts at 65,535 octets. `SETTINGS_INITIAL_WINDOW_SIZE` changes the initial stream window, including the effective window on active streams as specified there. It does not directly increase the connection window; a `WINDOW_UPDATE` on stream 0 does that. Implementations commonly tune these windows, so a capture's advertised values outrank the default in a diagnosis.

Here is an explanatory accounting model, not an extra equation stated by the RFC. If a sender has $S$ octets of available credit on a particular stream, $C$ octets of available connection credit, and $D$ octets of DATA ready to send, the currently permitted DATA on that stream is at most $\min(S,C,D)$. Sending $x$ DATA octets decreases both $S$ and $C$ by $x$. A `WINDOW_UPDATE` with increment $u$ increases the addressed window by $u$, subject to the protocol's limit. The sender must also respect TCP congestion and receive limits, so this application credit is an upper bound, not a promise of immediate delivery.

Consider a derived example. The stream and connection both start at 65,535 octets. The server sends 32,000 octets of response DATA on stream 1. Each has 33,535 octets of credit left. It then sends 30,000 octets on stream 3. Stream 3 retains 35,535 octets, but the connection retains only 3,535 octets. Even if stream 3 has plenty of individual credit and a response queued, it can send at most 3,535 more DATA octets before the connection window is refreshed. The arithmetic is `65,535 - 32,000 - 30,000 = 3,535`. A graph showing only per-stream windows would miss the shared ceiling.

Now imagine the receiver sends a `WINDOW_UPDATE` for stream 3 but not for stream 0. Its individual stream credit rises, while the connection credit remains 3,535 octets. Goodput does not improve. The inverse is also possible: the connection receives credit, but one stream remains exhausted. A tuning change that raises only `SETTINGS_INITIAL_WINDOW_SIZE` can therefore fail to fix a large transfer if the connection credit is still tight. The correct tuning target depends on which limit is binding in the actual trace. If the connection is shared by many streams, the connection window can become the bottleneck even when no individual stream consumes much.

The bandwidth-delay product gives a useful scale, provided we label it as a model. At a stable goodput $R$ bytes per second and a feedback round trip $T$ seconds, roughly $RT$ bytes must be allowed in flight to keep a pipe busy. This is not an HTTP/2 specification equation and does not include all buffering or scheduling delays. With an illustrative 20 Mbit/s link and 80 ms round trip, $R = 20{,}000{,}000 / 8 = 2{,}500{,}000$ bytes/s, so $RT = 2{,}500{,}000 \times 0.08 = 200{,}000$ bytes. A 65,535-octet window is about 0.328 of that modeled pipe volume. If updates arrive only after the sender has exhausted that window, the sender can spend time waiting for credit even though the link could carry more. An eager receiver can issue updates before exhaustion, so this simple ratio is a diagnostic prompt, not a universal throughput bound.

A second derived example puts the gRPC complaint in context. A stream carries a 1 MiB response, where 1 MiB is 1,048,576 bytes. If neither window were refreshed, only the first 65,535 DATA octets could cross under default credit. The remaining `1,048,576 - 65,535 = 983,041` octets need additional credit. That is a statement about the flow-control protocol, not a claim that every 1 MiB gRPC call pauses after exactly 65,535 bytes. Most implementations update windows as data is consumed and may advertise larger windows. A slow reader, middleware that buffers without draining, or a proxy with mismatched update behavior can expose the stop. A fast reader and proactive updates can hide it.

The operational question is therefore “who is consuming?” rather than “what is the configured number?” A receiver should replenish credit when it can safely accept more DATA, which often tracks application consumption and available memory. Advertise too little credit and you can underfill a high bandwidth-delay path. Advertise too much and a slow application may accumulate a larger amount of data in memory across active streams. A connection with hundreds of streams makes the aggregate concern more important. Throughput and memory protection are in tension; there is no globally correct magic window size.

The frame signature is concrete. In a verbose HTTP/2 log, record inbound `SETTINGS_INITIAL_WINDOW_SIZE`, connection and stream `WINDOW_UPDATE` frames, DATA lengths, and which stream ID stopped. A stream-specific stall affects one stream. A connection-credit stall can affect every stream sending DATA in that direction. TCP retransmissions without exhausted HTTP/2 credit point lower in the stack. An application that never enqueues DATA points higher. Inspect both directions because request-body upload and response-body download use independent receiver advertisements.

### The direction of credit is easy to reverse

When the server sends a response body, the client is the receiver of that DATA and controls the credit the server may spend. When the client uploads a request body, the server is the receiver and controls credit in the other direction. The endpoint that emits a WINDOW_UPDATE is granting the *peer* permission to send more DATA. This is easy to misread in a bidirectional gRPC stream because request and response messages are interleaved in time, and both sides can be blocked for different reasons at once. Label every window in a trace with the direction of DATA it permits: client-to-server upload or server-to-client download.

A proxy makes the direction question more subtle. A downstream client grants credit to the proxy, while the proxy grants credit to the upstream server. Those are different HTTP/2 connections with separate stream IDs and independent flow-control state. If the proxy accepts an upstream response into memory before sending it downstream, it can temporarily mask a slow client from the upstream server. Once the proxy's buffer fills, it may stop reading upstream or stop issuing upstream WINDOW_UPDATE. The original slow consumer then propagates backpressure across a connection boundary, but with delay and buffering. A trace on only the upstream side might make the proxy appear to be the slow receiver. A trace on only the downstream side might make the server appear slow. Correlate the two sides with request IDs and timestamps, while keeping each connection's window arithmetic separate.

The frame accounting also has a precise scope. HTTP/2 flow control applies to DATA payload octets, not to every byte carried by TCP. Frame headers, TLS records, TCP headers, and retransmitted bytes consume link capacity, but they are not deducted from the HTTP/2 window as if they were new DATA payload. This matters when comparing a calculated credit budget with interface byte counters. The two numbers answer different questions. If a link sends 1 MiB of wire traffic, that does not mean exactly 1 MiB of HTTP/2 flow-controlled DATA was accepted. Protocol overhead and retransmissions make wire bytes larger, while compression can make logical application data larger than encoded bytes.

Do not interpret a WINDOW_UPDATE as proof the application has completed work on the bytes. It says the endpoint has made receive capacity available according to its implementation. The endpoint might have copied data into another buffer, passed it to a runtime, or actually delivered it to application code. For gRPC, those distinctions can move the perceived stall between handler time and network time. A library may proactively replenish credit to keep the pipe full while messages queue for a slow application. That improves transfer progress but can increase memory. Another library may tie credit closely to application reads, exposing backpressure sooner. Both can comply with the protocol. To compare them, record buffer occupancy and consumer rate alongside frame credit, then state the library and version used in the experiment.

### Window arithmetic is not a throughput promise

It is tempting to calculate bandwidth-delay product and set every HTTP/2 window above it. That calculation identifies one necessary condition for keeping a path full under a simplified model. It does not prove the sender has data, the receiver can consume it, the TCP congestion window is large enough, or the scheduler serves the stream. It also does not say how often credit is replenished. A receiver that sends WINDOW_UPDATE before the available credit runs low may sustain high goodput with a smaller advertised window than a stop-and-wait mental model predicts. A receiver that batches updates conservatively may stall despite an apparently generous initial number.

There is another scaling dimension: multiple streams compete for one connection allowance. Suppose each of ten streams can have 200,000 DATA octets in flight by its stream credit, but the connection has only 200,000 octets of credit total. The aggregate is still bounded by 200,000, not ten times that number. If the connection allowance is raised to 2,000,000 octets, the receiver must be prepared for a correspondingly larger amount of unconsumed DATA in the worst case, depending on how promptly it reads and frees memory. These numbers are an illustrative calculation, not protocol defaults or a benchmark. They show why the stream setting and connection setting must be considered together, and why memory testing belongs beside throughput testing.

Finally, flow control is not an overload policy. It bounds bytes in transit to a receiver. It does not limit the rate at which a peer can start and cancel requests, the CPU cost of decoding headers, or the number of upstream operations already dispatched. Rapid Reset exploited precisely that gap. A service needs request admission, cancellation propagation, and work budgets in addition to byte credit. Treating a flow-control window as a DDoS throttle confuses a byte buffer limit with a request-rate limit.

## HPACK changes bytes, not the meaning of a header

![Repeated HTTP headers represented by HPACK table entries rather than repeated literal strings](/imgs/blogs/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer-4.webp)

HTTP/2 did more than split messages into frames. It introduced HPACK, a header compression scheme defined by [RFC 7541, May 2015](https://www.rfc-editor.org/rfc/rfc7541). HTTP requests routinely repeat field names and values. A browser may send the same authority, content type, or cookie on many requests to one origin. Repeating those strings in every header block consumes bytes and, on a constrained path, time. HPACK represents common fields using an indexed static table and repeated fields using a dynamic table maintained in encoder and decoder state. The wire representation changes; the HTTP header semantics do not.

Think of the dynamic table as a small shared notebook for one compression direction on one connection. The encoder can place an entry into that notebook. Later header blocks can refer to the entry by index rather than spell it again. The decoder must maintain corresponding state to interpret the index. The illustration is conceptual, not a byte-level HPACK encoding: a literal header becomes an indexed reference on a later request. Real HPACK encodings include integer representations, optional Huffman coding, and rules governing when fields are indexed. The RFC is the source for those exact formats.

This creates a diagnostic trap. A trace showing a tiny HEADERS frame does not mean the application sent almost no headers. A long cookie or metadata field can be represented compactly after table warm-up. Conversely, a privacy-sensitive field might be encoded without indexing, and the first occurrence of a value may still be large. A proxy can have separate HPACK state on its downstream and upstream connections. Comparing frame sizes on the two sides without accounting for separate compression contexts tells us little about the logical header list. Decode the HTTP/2 stream before diagnosing application metadata size.

The dynamic table is bounded by a peer-advertised size. The default `SETTINGS_HEADER_TABLE_SIZE` is 4,096 octets in [RFC 9113 section 6.5.2](https://www.rfc-editor.org/rfc/rfc9113#section-6.5.2), though peers can announce a different value. That number is a protocol setting, not a promise that the header block will be 4,096 bytes or less. The table stores reusable entries; a header list can be larger and has separate limits and operational controls. Repeated values compete for limited table space. If the pattern churns through many unique values, the encoder can spend effort updating a table that yields little reuse.

Compression also makes frame ordering significant. RFC 9113 requires a header block to be a contiguous sequence of `HEADERS` and `CONTINUATION` frames on the connection until it is complete. Interleaving another stream's frame into that unfinished header block is a protocol error. This is a local framing rule, distinct from the TCP head of line issue we discuss later. It is one reason the phrase “streams are independent” needs qualification. Streams have separate messages and flow-control windows, but they share connection state, header compression state, a connection window, and a transport.

HPACK was designed with security constraints that matter in operations. [RFC 7541 section 7](https://www.rfc-editor.org/rfc/rfc7541#section-7) discusses compression-based information leakage. Do not conclude that “smaller headers are always better” and indiscriminately index secrets. Most application teams should rely on a maintained HTTP/2 library's policy for sensitive headers, then inspect behavior if there is a specific leak or compression concern. When a proxy rewrites headers, it can also change which fields repeat, which affects compression ratio independently of user-visible request count.

If a gRPC call looks expensive on the wire, separate three sizes: serialized message bytes, decoded metadata bytes, and encoded HEADERS frame bytes. Protobuf changes message representation. HPACK changes header representation. Neither shrinks a DATA body by itself. The adjacent [API performance post](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) owns payload design and application compression decisions. Here, the useful measurement is the frame sequence and sizes on a named connection, plus the decoded headers when permission and privacy policy permit inspection.

## Priority is a hint whose deployment history matters

A service does not get a fair scheduler merely because it negotiates HTTP/2. RFC 9113 describes the original dependency and weight priority mechanism but deprecates the PRIORITY frame and the older signaling model. [RFC 9218, June 2022](https://www.rfc-editor.org/rfc/rfc9218) explains why: the dependency tree suffered limited deployment and interoperability, and the newer scheme expresses urgency and incremental delivery more directly. A client can signal preference, but the server chooses how to schedule its output. Intermediaries may translate, ignore, or replace signals. A packet capture of priority information is evidence that a preference was sent, not proof that it was honored.

Imagine six requests: one CSS file needed to render, one large image, and four small API reads. If the sender has queued data for all six and the socket is writable, it can choose which stream's frames to emit first. A priority scheme can express a desired ordering. But priority cannot make a missing TCP segment arrive, create connection flow credit, or force a slow application to produce bytes. Treat priority as a scheduling question only after confirming there is actually a scheduling choice. If one stream is the only stream with DATA ready, seeing it dominate the trace says nothing about fairness.

A misleading experiment is to compare total page load time after flipping a priority setting while also changing caching, connection reuse, origin placement, and payload sizes. The measured difference has too many possible causes. A useful experiment fixes the objects, connection topology, and impairment model, then records each response's first and last DATA frame times along with the priority signal. Even then, the result belongs to that client and server pair. It does not imply all HTTP/2 implementations behave the same way.

The practical failure mode is a large stream that appears to monopolize the connection. There are at least four possibilities. First, it may be the only stream with ready data. Second, the scheduler may give it too much service despite other ready streams. Third, another stream may be flow-control blocked, so the scheduler cannot send it. Fourth, TCP may be stalled, so no frame from any stream can advance. A trace of ready state is often inaccessible inside a vendor proxy, so combine wire evidence with per-stream application timestamps and window logs. If all streams pause at the same TCP sequence gap, changing an HTTP priority hint is the wrong lever.

## Six requests, three wire contracts

<figure class="blog-anim">
<svg viewBox="0 0 900 365" role="img" aria-label="Six identical requests under one HTTP/1.1 connection without pipelining, one HTTP/2 TCP connection, and one HTTP/3 QUIC connection. A lost packet stalls later bytes in the first two examples but only its QUIC stream in the third." style="width:100%;height:auto;max-width:900px">
<style>
.h2a-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:1.5}.h2a-t{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.h2a-s{font:13px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}.h2a-r{fill:var(--accent,#6366f1)}.h2a-loss{fill:#e8796b}.h2a-wait{fill:#e8b75a}.h2a-num{font:600 14px ui-sans-serif,system-ui;fill:#fff;text-anchor:middle}
@keyframes h2a-stall{0%,20%{opacity:0}30%,68%{opacity:1}78%,100%{opacity:0}}
@keyframes h2a-progress{0%,20%{opacity:.28}30%,68%{opacity:1}78%,100%{opacity:.28}}
.h2a-block{animation:h2a-stall 12s ease-in-out infinite}.h2a-live{animation:h2a-progress 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.h2a-block,.h2a-live{animation:none}.h2a-block{opacity:1}.h2a-live{opacity:1}}
</style>
<text class="h2a-t" x="14" y="26">Six requests, three wire contracts</text>
<text class="h2a-s" x="14" y="47">Same six requests; the shaded loss phase is an illustrative packet gap, not a speed ranking.</text>
<text class="h2a-t" x="14" y="101">HTTP/1.1</text><text class="h2a-s" x="14" y="120">one persistent TCP connection, no pipelining</text>
<text class="h2a-t" x="14" y="190">HTTP/2</text><text class="h2a-s" x="14" y="209">one TCP connection, interleaved frames</text>
<text class="h2a-t" x="14" y="279">HTTP/3</text><text class="h2a-s" x="14" y="298">one QUIC connection, independent streams</text>
<rect class="h2a-bg" x="365" y="72" width="510" height="68" rx="8"/><rect class="h2a-bg" x="365" y="161" width="510" height="68" rx="8"/><rect class="h2a-bg" x="365" y="250" width="510" height="68" rx="8"/>
<g class="h2a-live"><rect class="h2a-r" x="380" y="87" width="54" height="38" rx="6"/><rect class="h2a-r" x="448" y="87" width="54" height="38" rx="6"/><rect class="h2a-loss" x="516" y="87" width="54" height="38" rx="6"/><rect class="h2a-wait" x="584" y="87" width="54" height="38" rx="6"/><rect class="h2a-wait" x="652" y="87" width="54" height="38" rx="6"/><rect class="h2a-wait" x="720" y="87" width="54" height="38" rx="6"/></g>
<g class="h2a-live"><rect class="h2a-r" x="380" y="176" width="54" height="38" rx="6"/><rect class="h2a-r" x="448" y="176" width="54" height="38" rx="6"/><rect class="h2a-loss" x="516" y="176" width="54" height="38" rx="6"/><rect class="h2a-wait" x="584" y="176" width="54" height="38" rx="6"/><rect class="h2a-wait" x="652" y="176" width="54" height="38" rx="6"/><rect class="h2a-wait" x="720" y="176" width="54" height="38" rx="6"/></g>
<g class="h2a-live"><rect class="h2a-r" x="380" y="265" width="54" height="38" rx="6"/><rect class="h2a-r" x="448" y="265" width="54" height="38" rx="6"/><rect class="h2a-loss" x="516" y="265" width="54" height="38" rx="6"/><rect class="h2a-r" x="584" y="265" width="54" height="38" rx="6"/><rect class="h2a-r" x="652" y="265" width="54" height="38" rx="6"/><rect class="h2a-r" x="720" y="265" width="54" height="38" rx="6"/></g>
<g class="h2a-num"><text x="407" y="112">1</text><text x="475" y="112">2</text><text x="543" y="112">3</text><text x="611" y="112">4</text><text x="679" y="112">5</text><text x="747" y="112">6</text><text x="407" y="201">1</text><text x="475" y="201">2</text><text x="543" y="201">3</text><text x="611" y="201">4</text><text x="679" y="201">5</text><text x="747" y="201">6</text><text x="407" y="290">1</text><text x="475" y="290">2</text><text x="543" y="290">3</text><text x="611" y="290">4</text><text x="679" y="290">5</text><text x="747" y="290">6</text></g>
<g class="h2a-block"><text class="h2a-s" x="784" y="113">later wait</text><text class="h2a-s" x="784" y="202">later wait</text><text class="h2a-s" x="784" y="291">only 3</text></g>
<text class="h2a-s" x="365" y="342">Red: lost packet carrying request 3 bytes. Amber: delivery waits for recovery.</text>
</svg>
<figcaption>With the stated HTTP/1.1 and connection assumptions, one loss delays subsequent work; HTTP/2 still shares TCP ordering, while HTTP/3 confines transport loss to the affected QUIC stream. This illustrates isolation, not universal speed.</figcaption>
</figure>

The animation uses the same six logical requests and one loss event to keep the comparison honest. Its HTTP/1.1 lane assumes one connection with sequential responses for the purpose of illustrating application-layer ordering. A browser that opens several HTTP/1.1 connections can avoid that particular single-connection queue, at the price of more connection state and handshakes. Its HTTP/2 lane uses one TCP connection with interleaved frames; the responses can advance independently until one TCP segment is lost, then in-order delivery stalls them all. Its HTTP/3 lane uses QUIC streams. Loss of a packet containing one stream's data need not block delivery of data from other streams, although congestion control and shared capacity still affect the entire connection. This is a mechanism illustration, not a claim that one protocol always has lower page-load time.

The standards supply the boundaries. [RFC 9114, June 2022](https://www.rfc-editor.org/rfc/rfc9114) maps HTTP request exchanges onto QUIC streams. [RFC 9000, May 2021](https://www.rfc-editor.org/rfc/rfc9000) defines QUIC's stream transport and loss recovery behavior. In HTTP/2 over TCP, an HTTP/2 stream identifier is a label inside the ordered TCP byte stream. In HTTP/3 over QUIC, stream offset and reassembly are transport-level concepts. That difference lets a QUIC receiver deliver intact data for one stream while another waits for retransmission. It does not eliminate loss, retransmission cost, congestion response, or an application dependency between requests.

A real page can violate the independence assumed by the visual comparison. CSS may have to arrive before layout. An API call may need an authentication refresh. A JavaScript bundle may discover more resources only after execution. If those dependencies dominate, transport stream independence offers less benefit than the diagram suggests. Likewise, on a clean low-latency path where all six payloads fit quickly, loss isolation may never matter. Use the animation to predict a packet trace under its stated assumptions, not to rank protocols without measuring the workload.

The other tempting simplification is that HTTP/2 always uses exactly one connection. Browsers often coalesce requests to a connection under specific authority and certificate rules, while clients and libraries can open multiple connections. A proxy may terminate and re-originate. A gRPC channel may own one or more HTTP/2 connections depending on the implementation. The six-request experiment is a controlled model. Production diagnosis begins by counting actual connections, stream IDs on each, and the layer at which each is terminated. The [QUIC and HTTP/3 companion](/blog/software-development/networking/quic-and-http-3-what-moving-to-udp-actually-changed) goes deeper on how changing the transport alters connection behavior.

## The head of line moved into TCP

![One missing TCP sequence range delays delivery of HTTP/2 frames from several streams](/imgs/blogs/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer-5.webp)

The name *head-of-line blocking* describes a queue whose front item prevents later items from advancing. It is not a single defect with a single fix. HTTP/1.1 pipelining had an application protocol ordering constraint. HTTP/2 multiplexing removes that response ordering constraint by labeling frames with stream IDs. TCP still delivers an ordered byte stream. A missing sequence range blocks later received bytes from reaching the HTTP/2 parser until loss recovery fills the gap. We moved the blocking boundary down a layer.

Consider an illustrative capture at the receiver. TCP bytes through sequence offset (n) arrive. A segment covering the next range is lost. A later segment containing a complete-looking HTTP/2 frame for a different stream reaches the host. The kernel can acknowledge out-of-order data according to its TCP behavior, but an application read of the socket cannot jump over the missing sequence range. The HTTP/2 implementation has no frame to schedule or dispatch from those later bytes yet. The crucial measurement is not whether the later packet reached the NIC. It is whether the ordered TCP stream delivered its bytes to the parser.

This distinction can explain a strange set of traces. The sender reports that several stream frames were written. The receiver packet capture shows later segments arriving. Application logs show no corresponding response chunks for any stream until a retransmission. Nothing in that sequence requires slow application code. The wait belongs to TCP reassembly. A rising retransmission counter and a visible sequence gap corroborate it. A `WINDOW_UPDATE` trace may continue independently because control frames in the opposite direction can still travel; that does not mean response DATA has been delivered to the client application.

A loss event does not imply an entire round trip of delay in every case. TCP loss recovery has multiple paths, including duplicate acknowledgments and selective acknowledgment where supported, and timers for cases with insufficient subsequent data. The precise delay depends on the sequence pattern, sender and receiver stacks, RTT, and congestion state. Do not attach a universal millisecond penalty to “one lost packet.” Instead, measure the gap between the missing sequence range and successful retransmission at one capture point. Then compare that gap to the simultaneous pause across HTTP/2 stream IDs.

There is a second subtlety: a stream may appear unaffected even if a segment was lost on the same connection. If its entire response had already been delivered before the gap, it has nothing left to wait for. If application work delays its next bytes until after recovery, the transport stall may be hidden under server think time. The claim is about bytes later in the same TCP stream that are ready to be delivered while a hole exists, not about every logical request on the connection at every moment.

One mitigation is to use HTTP/3 on paths where independent stream delivery under loss matters, provided client and server support it and the workload benefits. Another is to avoid placing unrelated latency-critical work behind a single overloaded connection, though opening extra TCP connections has costs and can change load-balancer behavior. Neither replaces fixing packet loss or queueing on the path. If retransmissions rose after a deployment, investigate the path and host counters before treating a protocol switch as the root-cause repair. The [reliability post](/blog/software-development/networking/reliability-sequence-numbers-acks-retransmits-and-rto) explains TCP recovery in detail; the [bufferbloat post](/blog/software-development/networking/bufferbloat-queueing-and-the-latency-you-added-yourself) explains a different reason all streams can slow together without packet loss.

## A diagnosis that separates flow control from transport loss

![Decision tree from slow streams to window, retransmission, or server evidence](/imgs/blogs/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer-6.webp)

Start with the user-visible interval: which requests slowed, which direction was transferring bytes, and whether they shared one connection. Then record three aligned timelines: application readiness, HTTP/2 frame and window activity, and TCP sequence/retransmission activity. The figure is a decision tree, not a list of optimizations. Each branch should have an observable discriminator. If every stream stalls while retransmissions rise and a sequence gap appears, investigate transport. If the sender has DATA ready but the relevant flow-control window reaches zero, investigate receiver consumption and window policy. If neither credit nor transport blocks the bytes, investigate the application's production of data or the proxy's scheduling and stream admission.

A low-overhead first check on Linux is `ss -tin` for the connection and `nstat -az` for TCP counters, scoped by time and connection where possible. Counters are corroboration, not proof about a particular HTTP/2 stream. A host-wide retransmission count can rise because of unrelated traffic. A filtered packet capture provides sequence-level evidence but may contain headers and payloads, so use a narrow host and port filter, an appropriate capture length, and the organization's data-handling controls. `tshark` can decode HTTP/2 frames after TLS decryption only when key material or an approved termination-side trace is available. A plaintext h2c lab avoids that complication; production TLS should not be weakened just to make packet inspection convenient.

The `nghttp` client can print HTTP/2 frame exchange in a controlled environment. Watch for `SETTINGS`, the advertised initial window, `WINDOW_UPDATE` on stream 0 and on individual streams, and the DATA lengths around the stall. An absent update by itself does not prove a fault; the receiver may not yet have consumed enough bytes or may have advertised ample credit. Reconstruct the running credit and compare to queued DATA. A debugging log that says “flow control blocked” is useful, but the peer's frame trace establishes whether credit actually arrived. Remember that a proxy is two separate HTTP/2 endpoints. A downstream client can grant credit while the upstream proxy remains blocked by its own downstream or buffering policy.

For gRPC, correlate transport evidence with message consumption. A server streaming method can keep producing messages while the client stops reading. The gRPC library may buffer some messages before backpressure reaches the handler, so handler timestamps alone can understate the stall. Add timestamps for the write attempt, write completion, client read, and stream cancellation in a controlled reproduction. Do not log sensitive message contents merely to diagnose flow control. The sibling [gRPC on the wire](/blog/software-development/networking/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap) covers deadlines and load balancing; this post's question is which receive window admits its bytes.

A useful production runbook has four stages. Confirm HTTP/2 negotiation and actual connection sharing. Check whether the sender has DATA ready and whether the receiver has granted stream and connection credit. Check TCP retransmission and RTT changes during the same interval. Finally, inspect server and proxy scheduling only after the two explicit blocking mechanisms are excluded. This ordering prevents a common failure: tuning a server's stream limit because many calls are slow while the connection is actually in TCP loss recovery.

### Three trace patterns that look similar in a dashboard

Consider three illustrative traces with the same visible symptom: a caller receives a large response slowly. These are diagnostic examples, not measurements from a named service. In the first, the server has encoded DATA ready, the client's stream window reaches zero, and the client sends no stream-level WINDOW_UPDATE until its application reads another chunk. Other streams on the connection continue receiving DATA. The smallest blocked boundary is that stream's receiver. Changing TCP congestion control would be irrelevant. Inspect the client's read loop, message processing, and memory policy. If a callback does expensive work before asking the gRPC runtime for another message, the transport may be correctly applying backpressure. Raising its stream window could postpone the stall while allowing a larger buffer to accumulate, but the application still consumes at the old rate.

In the second trace, several streams stop sending DATA together. Each has positive stream credit, but the connection window has reached zero. The receiver issues a connection-level WINDOW_UPDATE, and all streams resume. The smallest blocked boundary is shared connection credit. This can happen even if every application consumer is reasonably fast: the aggregate of many streams spends the connection allowance before the receiver replenishes it. Check whether the client's runtime grants connection credit as it drains all streams, whether one stream is hoarding buffered bytes, and whether an intermediary has a different connection window on its upstream side. Increasing only per-stream initial windows cannot fix the shared limit. It might make the mismatch less obvious in a short test while leaving the aggregate behavior unchanged.

In the third trace, stream and connection credit are both positive, but a TCP sequence gap opens. The receiving host gets later packets, then delivers no later HTTP/2 frames to the parser until retransmission. A connection-level WINDOW_UPDATE may appear in the opposite direction, but it does not repair the missing sequence bytes. All streams with pending later bytes pause. The smallest blocked boundary is TCP ordered delivery. Investigate retransmissions, path loss, queue drops, and offload or capture artifacts before changing HTTP/2 settings. A packet capture at one host is an observation point, not proof of where loss occurred; sender and receiver captures may be needed to locate the missing segment.

The three traces can also overlap. A client may stop reading because its application is slow, then its receive buffers and HTTP/2 credit tighten while TCP also experiences loss. The discipline is temporal: identify which condition appeared first and which condition actually prevents the next DATA byte from reaching the caller. A zero window that appears after a long application pause may be an effect of the slow consumer. A retransmission that occurs after a flow-control stall may be unrelated background traffic. Compare stream IDs, packet sequence ranges, and timestamps at the same connection and direction. Host-wide counters alone cannot make that causal distinction.

There is a fourth pattern worth checking before touching the network. The server application has not produced the next response chunk. Both HTTP/2 windows are open and TCP has no segment to send. A long database query, lock wait, compression step, or intentional streaming interval can account for that gap. A transport trace is useful precisely because it can show that no DATA was queued onto the wire. The next probe belongs in application traces and server timing. When a proxy sits between the server and caller, perform this reasoning separately on each connection. The proxy may receive the upstream bytes promptly but delay the downstream frames because of its own window or scheduler.

These examples suggest a minimal evidence record for an incident. Save the negotiated protocol and connection tuple. For each affected stream, save its ID, request start, first response HEADERS, first DATA, last DATA, and close or reset. For each direction, save the latest stream and connection credit, plus TCP retransmission and RTT observations. Record the server's ready-to-write timestamp if available. Keep the time source and capture point in the record. A timeline with those fields is much more useful than a single “HTTP/2 p99” chart because it tells the next engineer which layer to inspect.

## Case study: Rapid Reset turned stream cancellation into work

The public case is the HTTP/2 Rapid Reset attacks Cloudflare began noticing on **August 25, 2023**, described in its [October 10, 2023 technical breakdown](https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/). Cloudflare, as the operator reporting the event, said the largest attack exceeded 201 million requests per second. That is its observed edge traffic under that attack, not a throughput estimate for ordinary HTTP/2, and it must not be applied to another provider or a normal service. The same write-up reports that its internal dashboard showed about 1 percent of requests affected during the initial wave, with brief peaks near 12 percent on August 29. Those percentages refer to Cloudflare's affected traffic and observation window, not to all users of HTTP/2.

The mechanism belongs exactly at the stream and connection boundary. An attacker sent a request `HEADERS` frame and immediately sent `RST_STREAM`, repeatedly on the same HTTP/2 connection. Open and half-closed streams count against a concurrency limit, but a reset quickly closes a stream and frees that slot. The sender can therefore churn through a large number of request starts without maintaining that many simultaneous open streams. If receiving and dispatching a request creates upstream work that takes longer to cancel, the expensive state can accumulate behind the proxy even though the visible HTTP/2 stream is already closed. The trigger was malicious rapid request-reset traffic. The contributing condition was how implementations handled cancellation and asynchronous work. The blast-radius multiplier was a small amount of connection and frame traffic causing much more downstream processing.

Cloudflare's trace example is particularly useful because it ties HPACK, stream IDs, and stream limits together. In the [same October 2023 analysis](https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/), it reports a test trace where the first HEADERS frame was 26 bytes and subsequent HEADERS frames were 9 bytes because of HPACK. Packet 15 contained 525 requests, reaching stream 1051, even though the server had advertised a maximum of 100 concurrent streams. These are values from that one published trace and test configuration. They demonstrate that a concurrency limit caps streams open at an instant, not the rate at which a client can open and cancel streams. Small encoded headers make the repeated action cheap on the wire; they do not make the server-side handler cheap.

The first mitigation lesson was not simply “lower the concurrency limit.” Cloudflare says it reduced its maximum stream concurrency to 64 during the response, then found that some clients assumed 100 before receiving the server's SETTINGS frame. Legitimate pages could send 100 requests early; the excess streams were reset, and an existing reset-count mitigation could close the connection. Cloudflare restored the limit to 100 after identifying that compatibility issue. The numbers 64 and 100 describe its response and observed client interaction, not a recommendation to set a universal limit. A defensive control must be evaluated against both abusive churn and normal startup behavior.

For an operator, this case transfers two disciplines. First, monitor request *start rate*, cancellation rate, and resource retirement separately from current stream count. A low active-stream gauge can coexist with enormous request churn and growing backend work. Second, make cancellation propagate promptly through proxies and handlers, but do not assume it can undo work already sent to a database or another service. Test the whole path. The edge may stop writing a response while an upstream call continues consuming CPU. Rate limits, work budgets, and cancellation-aware admission protect different boundaries. The companion [network attacks post](/blog/software-development/networking/network-attacks-you-will-actually-meet) places this within a wider defensive model.

Rapid Reset is not the transport head-of-line problem. It is a separate failure made possible by cheap stream creation and cancellation. It belongs here because it punctures the idea that multiplexing automatically makes a connection efficient and isolated. Stream state is cheap in some places and expensive in others. The protocol exposes a cancellation control, but the implementation decides how quickly work is cleaned up. Good diagnosis follows that work across the same boundary that normal flow control crosses.

## What to tune, and what a tuning change can break

A window setting is an admission policy for bytes. A concurrency setting is an admission policy for streams. A priority signal is a scheduling preference. A timeout is a deadline. None substitutes for the others. This distinction is valuable because all four can alter the apparent speed of a request in a benchmark. A larger connection window can remove a credit stall but increase buffered bytes. A higher concurrent-stream limit can admit more calls but increase memory and handler pressure. More aggressive priority can improve one resource while delaying another. A longer timeout can make a slow transfer finish but worsen user-visible wait and occupied resources.

The change review should name the failure it is intended to remove. “Raise the window” is too vague to test. “Reduce the fraction of server-to-client intervals in which connection credit is zero while response DATA is queued” is measurable. Record that fraction before and after, along with response completion, receiver memory, and TCP retransmissions. If completion improves but memory grows sharply, the new setting may be trading one bottleneck for another. If the zero-credit fraction was already negligible, the window change had no causal basis. If a second connection improves latency, count how many streams each connection carries and whether the L4 balancer now distributes them differently. The mechanism might be backend selection rather than TCP loss isolation.

Before changing a window, write down the measured RTT, goodput target, active-stream count, receiver consumption rate, and actual advertised credit. A rough bandwidth-delay product is a starting estimate, not the final setting. For example, if a 20 Mbit/s, 80 ms path suggests about 200,000 bytes in flight by the model above, moving from a 65,535-octet connection window to a value above 200,000 may remove one possible cap. It may also allow about three times as much unconsumed DATA on that connection before the peer needs a new update. With many connections, that memory exposure multiplies. Test with a representative slow consumer, not only a fast benchmark client.

Do not change two window levels at once during diagnosis. If you raise stream credit and connection credit together and throughput improves, you still do not know which was binding. The controlled lab below changes both to demonstrate that the negotiated values change, but a production tuning experiment should isolate them in separate runs after the relevant trace identifies the likely bottleneck. Record memory, p99, retransmissions, and per-stream completion, not only aggregate bytes per second.

If the issue is TCP loss, increasing HTTP/2 flow credit might increase the amount of later data waiting behind a missing segment. It does not remove ordered delivery. If the issue is a slow reader, opening a second connection may spread load but merely move memory and admission pressure. If the issue is one backend pinned by an L4 balancer, a larger stream limit may concentrate more RPCs there. The safe change follows the measured boundary. For system-level backpressure and overload policy, see [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure). The networking [senior mental model capstone](/blog/software-development/networking/the-senior-engineers-network-mental-model) connects this diagnosis to the other layers.

## Run it yourself

### Question

Can we see independent HTTP/2 streams and both levels of receive credit on one connection, and can changing the receiver-advertised windows change the frame trace without changing the response body? This experiment establishes the flow-control mechanism. It does not manufacture a universal throughput benchmark or simulate QUIC. For TCP loss behavior, use the separate controlled loss experiment in the [retransmission post](/blog/software-development/networking/reliability-sequence-numbers-acks-retransmits-and-rto).

### Preconditions

Run inside the Linux VM or host used by the [series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), with namespace `c` at `10.77.0.1`, namespace `s` at `10.77.0.2`, and veth interfaces `c0` and `s0`. The commands require `iproute2`, `nghttp` and `nghttpd` from nghttp2, `python3`, `grep`, `rg` (ripgrep), and privileges to enter namespaces. Record `nghttp --version` and `nghttpd --version` on your host; the command flags are documented by the [nghttp client manual](https://nghttp2.org/documentation/nghttp.1.html) and [nghttpd server manual](https://nghttp2.org/documentation/nghttpd.1.html), checked September 30, 2026. The cleartext `h2c` listener is confined to this lab namespace. It is not a production TLS configuration.

```bash
set -euo pipefail
ip netns list | grep -E '^(c|s) '
ip -n c address show dev c0
ip -n s address show dev s0
ip -n c route get 10.77.0.2
ip netns exec c ping -c 2 -W 1 10.77.0.2
ip -n c -s qdisc show dev c0
ip -n s -s qdisc show dev s0
nghttp --version
nghttpd --version
ip netns exec s ss -lnt '( sport = :8080 )'
```

The last command should show no preexisting listener on port 8080 for this experiment. If it does, stop and choose a fresh lab environment rather than killing an unrelated process. The namespace and interface checks should show the canonical addresses. The `ping` check proves reachability, not application throughput. The qdisc output records impairments left by earlier experiments; reset only lab-created qdiscs before comparing runs.

### Baseline

The following complete shell fragment creates one 4,456,448-byte file and runs a cleartext HTTP/2 server in namespace `s`. `nghttp` accepts the `-w` stream-window and `-W` connection-window options as bit counts: `2**N-1` octets, per its manual. The baseline requests two distinct paths over one invocation. Each contains the same file bytes. The log files are diagnostic output and stay under `netlab/out/`.

```bash
set -euo pipefail
SLUG=http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer
OUT=netlab/out/$SLUG
mkdir -p "$OUT/root"
python3 - "$OUT/root" <<'PY'
from pathlib import Path
import sys
root = Path(sys.argv[1])
payload = b'http2-window-lab\n' * 262144
(root / 'a.bin').write_bytes(payload)
(root / 'b.bin').write_bytes(payload)
print('bytes per object', len(payload))
PY
ROOT=$(realpath "$OUT/root")
ip netns exec s nghttpd --no-tls -a 10.77.0.2 -d "$ROOT" -v 8080 >"$OUT/server.log" 2>&1 &
SERVER_PID=$!
printf '%s\n' "$SERVER_PID" >"$OUT/server.pid"
for i in 1 2 3 4 5; do
  if ip netns exec s ss -lnt '( sport = :8080 )' | grep -q 8080; then break; fi
  sleep 1
done
ip netns exec c nghttp -v -n -w 16 -W 16 \
  http://10.77.0.2:8080/a.bin \
  http://10.77.0.2:8080/b.bin >"$OUT/baseline.log" 2>&1
rg 'SETTINGS|WINDOW_UPDATE|DATA|stream_id=' "$OUT/baseline.log" | sed -n "1,80p"
```

Read the client's received `SETTINGS` and `WINDOW_UPDATE` frames, its sent settings, the DATA stream IDs, and the number of DATA octets reported in verbose lines. Expect at least two response stream IDs and repeated DATA activity. The advertised starting values for both receiver windows are 65,535 octets with `-w 16 -W 16`, because `2**16 - 1 = 65,535`. The exact count and timing of updates depends on this nghttp2 build and how quickly it consumes data. The file construction prints `len(payload)`, which is 17 bytes per repeated line times 262,144 repetitions, or 4,456,448 bytes per object; this derived size is larger than either starting window, so credit must be refreshed for completion. The printed size is the reproducible input size.

### Apply one change

Keep the files, server, namespace, and path the same. Change only the client-advertised receive-window options. In a separate production test, isolate stream and connection windows one at a time.

```bash
set -euo pipefail
SLUG=http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer
OUT=netlab/out/$SLUG
ip netns exec c nghttp -v -n -w 20 -W 20 \
  http://10.77.0.2:8080/a.bin \
  http://10.77.0.2:8080/b.bin >"$OUT/larger-window.log" 2>&1
```

### Compare

```bash
set -euo pipefail
SLUG=http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer
OUT=netlab/out/$SLUG
for name in baseline larger-window; do
  printf '%s\n' "$name"
  rg 'SETTINGS|WINDOW_UPDATE|DATA|stream_id=' "$OUT/$name.log" | sed -n "1,80p"
  rg -c 'recv DATA frame' "$OUT/$name.log" || true
done
```

Read the startup `SETTINGS_INITIAL_WINDOW_SIZE`, connection-level `WINDOW_UPDATE`, each response stream ID, and subsequent window updates. With `-w 20 -W 20`, the advertised target is `2**20 - 1 = 1,048,575` octets, subject to the command's exact nghttp2 behavior. Both runs should complete the same response bodies. Expect the larger initial credit to require fewer early updates or move the first update later; the exact number can vary with library thresholds and DATA fragmentation. On a local veth with almost no RTT, elapsed time may be indistinguishable. That is not a failed lab: the falsifiable claim concerns credit and frame state. If you add controlled latency in a later experiment, print the qdisc configuration and confirm RTT with `ping` before comparing throughput.

### Reset

```bash
set -euo pipefail
SLUG=http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer
OUT=netlab/out/$SLUG
if test -f "$OUT/server.pid"; then
  kill "$(cat "$OUT/server.pid")" 2>/dev/null || true
  rm -f "$OUT/server.pid"
fi
```

This removes only the listener started by this lab. It leaves generated files under `netlab/out/` for inspection and does not delete namespaces or qdiscs shared with other experiments. On a production host, prefer read-only `ss -tin`, application transport metrics, and approved, filtered captures. Entering namespaces and packet capture may require root or Linux capabilities. Captures can contain credentials and application payloads; do not copy them into an issue without review.

## Key takeaways

HTTP/2 turns one connection into several logical streams with explicit frames. That removes HTTP/1.1 response ordering on the connection, and HPACK can reduce repeated header bytes. Neither feature makes a stream an independent transport connection. DATA consumes both stream and connection receive credit, while all HTTP/2 frames still ride one ordered TCP byte stream. A stalled gRPC response can therefore be a slow consumer, a shared connection-window ceiling, missing TCP bytes, a scheduler choice, or server work. Count actual connections and stream IDs, reconstruct credit, then inspect TCP sequence recovery before changing knobs. The best protocol setting is the one that addresses the measured boundary without moving the bottleneck into memory or upstream work.

## Further reading

- [RFC 9113: HTTP/2, June 2022](https://www.rfc-editor.org/rfc/rfc9113), for framing, stream states, flow control, and the deprecation of the original priority machinery.
- [RFC 7541: HPACK, May 2015](https://www.rfc-editor.org/rfc/rfc7541), for the static and dynamic tables and security considerations.
- [RFC 9218: Extensible Prioritization, June 2022](https://www.rfc-editor.org/rfc/rfc9218), for deployment lessons and the newer priority signal.
- [Cloudflare's Rapid Reset technical breakdown, October 2023](https://blog.cloudflare.com/technical-breakdown-http2-rapid-reset-ddos-attack/), for the dated operator trace and mitigation caveat.
