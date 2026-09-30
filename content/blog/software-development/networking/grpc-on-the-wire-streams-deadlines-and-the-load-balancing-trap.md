---
title: "gRPC on the wire: Streams, deadlines, and the load balancing trap"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Trace a gRPC call through HTTP/2, carry its deadline across services, and diagnose why a connection-level balancer leaves backends uneven."
tags:
  [
    "networking",
    "distributed-systems",
    "grpc",
    "http-2",
    "rpc",
    "load-balancing",
    "deadlines",
    "cancellation",
    "keepalive",
    "service-mesh",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap-1.webp"
---

The service has four healthy replicas. One burns CPU while the other three wait. The client reports that it has called the Service VIP thousands of times. Its dashboard counts requests, not connections, so the picture looks impossible. Then somebody restarts the client and the hot replica changes. That is the clue: an L4 load balancer chose a backend for a TCP connection, and gRPC put the subsequent RPCs on that connection as HTTP/2 streams.

![Path map from gRPC client through the L4 balancer to a selected backend](/imgs/blogs/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap-1.webp)

The diagram above is the mental model. Name resolution tells the client which address to dial; application bytes do not flow through the resolver. An L4 balancer sees a transport flow, not each HTTP/2 stream inside it. A proxy that terminates HTTP/2 can make a different routing decision, but then its own upstream connection pool becomes another place to inspect. We will trace the wire format, carry a deadline across a service chain, separate cancellation from rollback, and build a small experiment that makes connection affinity visible. For the wider path from DNS to response, start with [what happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The companion [HTTP/2 post](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer) owns the transport mechanics in more depth.

## 1. Place gRPC on the request path

An RPC is a call with request and response semantics. A gRPC *channel* is the client's logical handle for reaching a target. The gRPC resolver discovers addresses; a load balancing policy chooses a subchannel for each call; a subchannel uses an HTTP/2 connection to carry one or more streams. A stream is a bidirectional sequence of HTTP/2 frames associated with one stream ID. A gRPC message is application data framed *inside* DATA frames. Those four nouns are different scopes. The [gRPC team's 2018 transport explanation](https://grpc.io/blog/grpc-on-http2/) distinguishes channel, RPC, and message, and the [gRPC load balancing design](https://github.com/grpc/grpc/blob/master/doc/load-balancing.md) explicitly places the per-call decision between resolution and the subchannel.

This layering changes the meaning of a request count. Ten thousand RPCs can travel through one channel and one TCP connection. The L4 device may see exactly one connection setup and one long lived flow. That device can be perfectly even over *connections* and badly uneven over *work*. If there are four backends and one very busy client connection, the chosen backend receives all that client's calls. If there are many comparable client connections, connection-level distribution can become adequate, but that is a workload property, not a gRPC guarantee.

We should also stop treating the channel as a guaranteed single socket. The [gRPC performance guide](https://grpc.io/docs/guides/performance/) says a channel uses zero or more HTTP/2 connections; a connection normally has a concurrency limit, and extra RPCs can queue at the client when that limit is reached. Resolver updates, reconnection, policies, and implementation details can change the connection set. The trap is narrower and more precise: **when many RPC streams share one established transport connection behind a connection-level balancer, that balancer cannot independently place those streams**.

The path map matters during diagnosis. At the client, inspect resolved endpoint addresses and the gRPC policy. At an L4 hop, count established connections by backend, not only requests. At an L7 proxy, inspect both downstream stream distribution and upstream connection reuse. At the app, count actual RPCs, status codes, and CPU. The location where imbalance first appears identifies the owner of the routing decision. An average across all replicas hides it.

There is a higher-level question of where load balancing belongs in an architecture. [Load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) owns that design trade-off. Here we care about what the device can see on a concrete flow and which measurement proves the selection granularity.

## 2. One RPC on the wire is not one packet

![Nested channel, RPC stream, message, and HTTP/2 frame scopes](/imgs/blogs/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap-2.webp)

The [gRPC over HTTP/2 protocol document](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md) is the source for the framing rules in this section, checked on 2026-09-30. A normal unary request begins with HTTP/2 HEADERS. The method is `POST`; `:path` names the service and method; `content-type` identifies gRPC; `te: trailers` helps detect incompatible intermediaries. If the caller supplied a deadline, `grpc-timeout` carries the remaining duration. The request message follows in DATA frames. A response normally has response HEADERS, zero or more length-prefixed messages in DATA, and trailing HEADERS that include `grpc-status`. A quick error can use trailers-only.

That last distinction saves time in incidents. HTTP status `200` alone does not mean the RPC succeeded. A server can send `:status: 200` and later report a gRPC error in `grpc-status`. A proxy that drops trailers or rewrites the response can turn a meaningful application status into a protocol error. When you inspect an HTTP/2 trace, read both the opening headers and the final trailers. The `grpc-status` field is the outcome; the HTTP status tells you whether the HTTP exchange was delivered in a form the gRPC peer could parse. The exact response rules are in the [gRPC protocol document](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md).

Each message has a five-byte gRPC prefix: one compression flag byte plus a four-byte unsigned big-endian message length. The payload bytes follow. This is a message boundary inside the byte stream of HTTP/2 DATA. HTTP/2 frame boundaries need not coincide with gRPC message boundaries, as the protocol document says. A single message can cross several DATA frames; a DATA frame can hold bytes from more than one message. Neither the five-byte prefix nor the DATA frame header is a protobuf field. They are transport framing. The payload might be protobuf, but the gRPC wire protocol does not make protobuf mandatory.

A worked framing example keeps the scopes straight. Suppose an uncompressed serialized request is 100,000 bytes. Its gRPC message envelope is 100,005 bytes, derived as 1 + 4 + 100,000. Using the HTTP/2 default maximum frame payload of 16,384 bytes, the message bytes need at least seven DATA frames because six carry at most 98,304 bytes and leave 1,701 bytes. This is a lower bound before any other frame sizing choice, flow-control pauses, TLS records, TCP segmentation, or retransmission. It does **not** imply seven packets. A packet capture at a TCP hop can show a different segmentation from an HTTP/2 decoder because the layers have different boundaries. RFC 9113, published June 2022, specifies the [HTTP/2 frame size and flow-control rules](https://www.rfc-editor.org/rfc/rfc9113).

For a small request, the fixed five bytes matter more proportionally. A 20-byte serialized message becomes 25 bytes before HTTP/2 headers, TLS, and TCP/IP. The prefix is 25 percent of the 20-byte payload, or 20 percent of the 25-byte framed message, depending on the denominator. State the denominator when quoting an overhead percentage. Header compression can reduce repeated metadata, but it does not remove the per-message prefix. Protocol buffers and API contracts are handled in [gRPC and Protocol Buffers](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming); here the point is what a capture or proxy must parse.

The connection also has two credit systems above TCP. HTTP/2 tracks flow-control credit for each stream and for the connection. DATA consumes that credit; HEADERS, PING, and other control frames do not. RFC 9113 gives 65,535 octets as the initial window for a new stream and for a connection, then lets implementations update the windows. The deployed window is a negotiated, changing value, not a permanent 64 KiB throughput ceiling. If a receiver stops consuming messages, a sender can stall on stream credit. If aggregate DATA exhausts connection credit, otherwise independent streams share that stall. This is why a write returning from a gRPC API need not mean bytes are already on the wire; the [gRPC flow-control guide](https://grpc.io/docs/guides/flow-control/) describes framework buffering and possible blocking. Do not confuse this receiver protection with TCP congestion control, which reacts to path capacity. [Flow control versus congestion control](/blog/software-development/networking/flow-control-vs-congestion-control-two-windows-one-pipe) gives the wider distinction.

Headers have their own limits. The [gRPC wire document](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md) suggests an 8 KiB default for a server's request-header limit and for a client's response-header and trailer limits, while allowing implementations to choose limits. That suggestion is not the size of a protobuf message and is not a universal deployed constant. A large authentication token, tracing baggage, or binary metadata field can fail before a handler sees the payload. A large error detail in `grpc-status-details-bin` can fail at the other end after the business operation already completed. In a proxy chain, every hop may apply its own header-list limit, so the first rejecting hop matters. Capture or log header sizes at the two sides of the hop where the outcome changes, with credentials redacted. Increasing a message-size limit cannot repair a header-list rejection.

There is a subtle observability trap here. Many HTTP metrics count the response when headers arrive. gRPC's meaningful final status usually arrives later in trailers. A proxy that reports the opening HTTP status as success while ignoring trailers can label failed RPCs as successful. A server might have spent most of the request deadline producing a message, then emit a nonzero gRPC status when a downstream dependency fails. If you graph only HTTP `200`, that whole failure mode disappears. Build RPC-level metrics from the gRPC status code and method at the endpoint or an aware L7 proxy, while retaining HTTP status as a separate transport signal. The protocol document permits a trailers-only error response too, so a trace parser must handle both forms.

The concurrency limit is yet another quantity. HTTP/2 `SETTINGS_MAX_CONCURRENT_STREAMS` restricts how many streams the peer permits at once. When active RPCs reach the configured limit, the gRPC performance guide notes that additional calls may wait in a client queue. A long-lived streaming RPC occupies a stream for its lifetime. Increasing a server's worker count cannot remove a queue that sits in the client before headers leave the process. Inspect active streams and client-side pending calls before blaming server CPU.

This also explains why multiplexing should not be sold as unlimited parallelism. Streams share a TCP connection, its congestion response, its connection-level HTTP/2 flow credit, and often a bounded number of simultaneous stream slots. If one packet is lost, TCP delivers later bytes in order, so unrelated streams can wait behind that missing segment even though HTTP/2 gives them separate IDs. The HTTP/2 post traces that head-of-line effect in detail. For this gRPC diagnosis, the practical point is to measure where waiting occurs: pending before HEADERS, blocked by flow credit after DATA starts, or delayed by transport loss. Those wait states can all inflate the same RPC latency histogram but have different fixes.

## 3. A deadline is a shrinking budget, not a hop-local timer

![Deadline budget shrinking from the first caller to a downstream service](/imgs/blogs/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap-3.webp)

The caller owns an end-to-end wait budget. If the client gives the whole operation 300 ms and the first service spends 80 ms before calling the second, the downstream call cannot honestly claim the original 300 ms. Its remaining budget is at most 220 ms before network transit, queueing, and cleanup reserve. This is an illustrative budget calculation: 300 ms − 80 ms = 220 ms. No production percentile is implied. If a client then gives each hop a fresh 300 ms, a chain can outlive the caller's intent. The user has already left while servers continue work that no response can use.

The [gRPC deadline guide](https://grpc.io/docs/guides/deadlines/), last modified July 2025, defines a deadline as the point after which the client is unwilling to wait. It says there is no deadline by default and recommends setting a realistic one. A language API might accept an absolute deadline or a relative timeout. On the wire, the gRPC protocol uses `grpc-timeout`, a duration with a numeric value and unit such as `250m` for 250 milliseconds. The receiving implementation should reduce the duration for elapsed time during propagation rather than forwarding an absolute wall-clock timestamp, which avoids skew between host clocks. Java and Go normally propagate deadlines; C++ needs explicit enablement according to the guide. Check the language and version before assuming your interceptor does it.

An explanatory budget model is useful for reviewing call graphs:

$$B_{child} = \max(0, B_{parent} - E - R)$$

Here $B_{parent}$ is the parent's remaining duration when work begins, $E$ is elapsed local work before the child call, and $R$ is a deliberate reserve for the response path and cleanup. This is a design model, not an equation mandated by gRPC. Implementations may encode and round timeout units differently. Its lesson is that the child gets the remaining budget, not a new full budget. For a parallel fan-out, each child can share the same absolute parent deadline; adding their durations as if they run serially is wrong. For a serial chain, each hop consumes part of the same budget.

Consider a second worked example. The entry point has 500 ms left. Authentication consumes 35 ms, the app queues for 90 ms, and response transport plus serialization is budgeted at 25 ms. The next service has at most 350 ms by subtraction: 500 − 35 − 90 − 25. If its own expected work is 400 ms, starting it is likely wasted under these assumptions. The safe decision might be a fast failure, degraded response, cache lookup, or a changed feature path. Raising all timeouts to 2 seconds would hide the budget mismatch while increasing the duration of abandoned work. This is why deadline tuning needs latency distributions and capacity tests, not a magic constant.

An expired deadline does not prove the server did nothing. The [gRPC deadline blog of February 26, 2018](https://grpc.io/blog/deadlines/) explicitly warns that a server can finish and send a response after the client has locally decided `DEADLINE_EXCEEDED`. For mutations, use idempotency keys or an application-level result lookup when ambiguity matters. Do not retry a write solely because the client saw a timeout. The gRPC transport can tell you that the caller stopped waiting; it cannot roll back a database commit across a service boundary.

In Go, pass the incoming context when making a child RPC so the deadline and cancellation remain attached. A separate context with `context.Background()` severs that chain. The following is an idiomatic fragment, not a complete server. `downstream` is an initialized generated client and the request types come from its protobuf package.

```go
func (s *Server) GetQuote(ctx context.Context, req *pb.QuoteRequest) (*pb.QuoteReply, error) {
    if err := ctx.Err(); err != nil {
        return nil, status.FromContextError(err).Err()
    }
    childCtx, cancel := context.WithTimeout(ctx, 250*time.Millisecond)
    defer cancel()
    price, err := s.downstream.GetPrice(childCtx, &pb.PriceRequest{Sku: req.Sku})
    if err != nil {
        return nil, err
    }
    return &pb.QuoteReply{Price: price.Value}, nil
}
```

`context.WithTimeout(ctx, ...)` can tighten the budget for this child. It cannot extend an earlier parent deadline. In a real handler, instrument the remaining deadline at the call site and the RPC status at completion. Log the deadline duration only with the request or trace context needed to interpret it. A naked `DEADLINE_EXCEEDED` count does not tell whether time was spent waiting for a client-side stream slot, traversing a proxy, queueing at the backend, or doing handler work. That distinction belongs to the same [diagnostic ladder](/blog/software-development/networking/a-diagnostic-ladder-for-network-problems) used across this series.

## 4. Cancellation is a signal, not a transaction rollback

Cancellation answers a different question from the deadline. A deadline says how long the caller will wait. Cancellation says the caller no longer wants this particular result, perhaps because a browser disconnected, a competing request won, or the application changed course. Deadline expiry and I/O failure can trigger cancellation too. The [gRPC cancellation guide](https://grpc.io/docs/guides/cancellation/), last modified February 2024, says the server should stop ongoing computation and propagate the cancellation to downstream work. It also says the library cannot generally interrupt an application handler that ignores its context. The network signal only becomes resource savings if the handler observes it.

At the HTTP/2 layer, a peer can end one stream with `RST_STREAM` while the shared connection remains available for other streams. That is the right mental picture for cancelling one RPC: it is not equivalent to dropping the TCP connection that carries many other RPCs. The details of stream states and reset handling are specified by [RFC 9113, June 2022](https://www.rfc-editor.org/rfc/rfc9113). A connection failure is broader: every active stream on that connection is affected. This difference matters when a dashboard lumps `CANCELLED`, `DEADLINE_EXCEEDED`, and `UNAVAILABLE` into one generic error bucket. They lead you toward different layers.

Consider a search endpoint that fans out to catalog, stock, and pricing. The caller cancels after receiving enough partial information to render a useful result. If the parent handler returns but its three worker goroutines continue with detached contexts, the user sees a fast response while backend load stays high. Under load, the abandoned work can queue behind new useful work. The next cohort's latency grows, causing more cancellations. The cycle can masquerade as a network regression even if packet loss stays flat. Propagating the parent context and checking it before expensive work interrupts that feedback loop. This is a causal example, not a claim about any named deployment.

Cancellation does not undo side effects already committed. The [gRPC core concepts guide](https://grpc.io/docs/what-is-grpc/core-concepts/) states this explicitly. If a payment handler writes to a database and then sees a cancelled context, returning `CANCELLED` cannot erase the write. If the client retries, the second attempt needs an idempotency key or a stable operation ID. This is the same ambiguity as deadline expiry: the client knows its own wait ended, not the global state of a distributed operation. For read-only work, stopping quickly is often safe. For writes, deciding whether to finish or compensate is an application policy built above gRPC.

When an incident shows many cancellations, inspect three times: when the client decided to cancel, when the server observed the signal, and when expensive handler work actually stopped. If the gap from observation to stop is long, inspect the handler's context checks and child RPC propagation. If the server never sees a cancellation, inspect intermediaries and client API usage. If `DEADLINE_EXCEEDED` is common before headers leave the client, the bottleneck might be a stream slot or connection pool, not server compute. These are falsifiable branches, not a generic instruction to raise timeout values.

The [status-code reference](https://grpc.io/docs/guides/status-codes/) names `CANCELLED` and `DEADLINE_EXCEEDED` separately, while the [error handling guide](https://grpc.io/docs/guides/error/) shows that transport failures may surface as `UNAVAILABLE`. Record a structured status code, the per-hop elapsed time, and whether headers were sent. The exact returned code can depend on where the race resolves, so do not infer a complete chronology from one status string alone. A packet capture or HTTP/2 frame log can settle whether a stream reset, connection close, or late trailers occurred.

## 5. Keepalive pings solve a connection question

A long-lived connection can be healthy at both endpoints and still disappear in the middle. A proxy or load balancer may close an idle transport to reclaim state. The next RPC discovers the close during reuse, reconnects, or fails depending on timing and implementation. That symptom often appears as a spike on the first call after quiet periods, while steady traffic behaves normally. First find the middlebox that owns the idle timeout. Then decide whether the connection should be kept alive or allowed to close cleanly.

gRPC keepalive uses HTTP/2 PING frames. A PING is connection-level and is acknowledged by the HTTP/2 peer; it is not a business RPC and does not exercise the application handler. It can test that the path and peer connection remain responsive. It cannot prove the service is ready to serve a particular method, so it is distinct from health checking. The [gRPC keepalive guide](https://grpc.io/docs/guides/keepalive/), last modified November 2025, makes this separation explicit and discusses proxies that consider quiet connections idle.

There are two timers to reason about. The ping interval decides how long the connection may be quiet before the client asks for an acknowledgment. The ping timeout decides how long it waits for that acknowledgment before treating the connection as broken. Neither is an application deadline. A 20-second ping timeout should not become a 20-second allowance for every RPC; the caller's useful deadline could be much shorter. Conversely, a 200-millisecond RPC deadline does not mean the client should emit PINGs five times per second. These knobs control different failure domains.

The current gRPC guide lists a disabled client keepalive interval by default, a 20-second keepalive timeout, and a server minimum permitted ping interval of five minutes in its generic option table. Those are guide defaults as checked on 2026-09-30, not universal values for every language, service, or managed proxy. Some servers disallow pings with no active calls; too-frequent pings can elicit a `GOAWAY` with `too_many_pings`. The guide recommends avoiding client pings much below one minute and coordinating with the service owner. A value that is smaller than an idle timeout is necessary for one keepalive design, but it is not sufficient: the server must accept it and every intermediary must pass it in the way the deployment expects.

For example, suppose a documented proxy idle threshold is 120 seconds and the server permits a 90-second interval. A 90-second ping may keep an otherwise idle connection active, leaving 30 seconds of margin under the threshold. These values are a hypothetical arithmetic example. If the proxy closes after 60 seconds instead, a 90-second ping arrives too late. If the server rejects pings more frequent than five minutes, choosing 30 seconds to beat the proxy can simply trade idle closes for `GOAWAY`. That conflict cannot be repaired with a local client flag alone. Change the proxy policy, permit a coordinated interval, or accept reconnects and make the first request robust.

Connection-level keepalive can also preserve an unwanted affinity. If one hot client connection remains alive indefinitely behind an L4 balancer, its RPCs remain assigned to the original backend indefinitely. This does not mean disabling keepalive is a sound load balancing strategy. Churning connections to distribute RPCs buys another selection at the cost of handshakes, TLS work, and churn in every intermediary. It is a diagnostic clue and sometimes a temporary workaround; deliberate per-call routing is the durable answer when a small set of long-lived connections carries uneven work.

The TCP stack has its own keepalive and timeout controls, but an HTTP/2 PING can traverse a TCP proxy where a local TCP probe only proves the client-to-proxy leg. The [gRPC guide's TCP_USER_TIMEOUT section](https://grpc.io/docs/guides/keepalive/) calls out that distinction. In a capture, identify the actual frame and hop before claiming the end-to-end path was tested. On a TLS connection, a passive capture without decryption cannot simply read the HTTP/2 PING inside the encrypted record. Use endpoint debug logging or a controlled cleartext lab for frame-level evidence.

## 6. Message size is separate from frame size and flow credit

The five-byte gRPC prefix can encode a large length, but the receiver does not have to accept an arbitrarily large message. A language runtime, server option, proxy, or API contract can impose a lower practical limit. In gRPC Go, the [package documentation for `MaxRecvMsgSize`](https://pkg.go.dev/google.golang.org/grpc#MaxRecvMsgSize), checked on 2026-09-30, says the default maximum receive message size is 4 MB when the option is unset. That is a Go implementation default. It is not an HTTP/2 standard limit, and it does not promise all languages or deployments use the same number. Configure both sending and receiving sides deliberately if your contract genuinely needs larger messages.

The units matter. A 4 MiB binary payload is 4,194,304 bytes. In decimal MB it is about 4.19 MB. If a limit is documented as `4MB`, inspect the implementation's actual byte constant or behavior before assuming whether the vendor means $4\times10^6$ or $4\times2^{20}$. The Go documentation's wording alone is not a license to assert a cross-language byte threshold. Use a controlled test payload near the limit and record serialized size, not the source object size or JSON representation. Compression also complicates the comparison: the wire payload may shrink while decompression expands into a larger message in memory.

Frame size, flow control, stream concurrency, and message size are four different limits. A 2 MiB message can be legal for the receiver yet require many HTTP/2 DATA frames. It can pause while stream or connection credit is refreshed. A hundred such messages in distinct RPCs can queue because concurrent stream slots are exhausted. Enlarging only `MaxRecvMsgSize` cannot solve flow stalls or stream-slot waits; enlarging only the HTTP/2 window cannot make an oversized application message valid. The layer at which the error first appears tells you which knob is relevant.

Large unary messages also increase the memory and latency cost of retries. If a caller sends a 16 MiB request, buffers it at one or more proxies, and retries it after a timeout, the path may hold multiple copies even before the backend decodes it. The 16 MiB is an illustrative size, not a claimed default. For a stream of records, chunk boundaries can bound per-message memory while preserving one logical operation. That does not eliminate backpressure: the producer must still read the receiver's pace. The [gRPC flow-control guide](https://grpc.io/docs/guides/flow-control/) warns that writes may wait and that unbalanced synchronous read/write patterns can deadlock under manual control. If records can be processed incrementally, a streaming API usually makes the memory and failure behavior more legible than one enormous unary blob.

Do not convert every unary method to a long-lived stream for speed. The [gRPC performance guide](https://grpc.io/docs/guides/performance/) says a stream can avoid setup per RPC but cannot be rebalanced after it begins, and long-lived streams complicate failures. That is a real trade-off in this post: a stream that holds a backend for hours is harder to drain or redistribute. If the application needs one ordered logical conversation, streaming can be appropriate. If it needs independent short operations, individual RPCs expose boundaries where the client or an L7 proxy can choose again.

When investigating `RESOURCE_EXHAUSTED`, collect the failing method, serialized request and response size, configured receive and send limits at both peers, proxy limits, and compression state. Avoid guessing from a packet count. A network trace sees compressed TLS records or DATA bytes; the runtime checks a decoded message. [API performance and payload size](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) covers broader payload design; this post keeps the diagnosis at the wire boundary.

## 7. The L4 trap: fair connections are not fair RPCs

Now return to the hot replica from the opening. The client resolves a Service VIP and opens one long-lived HTTP/2 connection. The L4 balancer chooses backend A for that connection. The client then opens stream 1, stream 3, stream 5, and so on over the selected transport. The balancer sees bytes on the established flow. It has no HTTP/2 stream parser at this layer and no opportunity to pick B for stream 3. Backend A's request counter rises. Backend B's connection and request counters remain flat. More healthy replicas do not change that mapping until another connection is selected.

<figure class="blog-anim">
<svg viewBox="0 0 760 290" role="img" aria-label="Successive RPC streams on one established TCP connection reach the same L4 selected backend" style="width:100%;height:auto;max-width:850px">
<style>
.grpc4-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.grpc4-label{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.grpc4-small{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.grpc4-line{stroke:var(--border,#d1d5db);stroke-width:3;fill:none}
.grpc4-dot{fill:var(--accent,#6366f1)}
@keyframes grpc4-left{0%{transform:translate(0,0);opacity:0}8%{opacity:1}44%{transform:translate(445px,0);opacity:1}49%,100%{transform:translate(445px,0);opacity:0}}
@keyframes grpc4-right{0%,50%{transform:translate(0,0);opacity:0}58%{opacity:1}94%{transform:translate(445px,0);opacity:1}100%{transform:translate(445px,0);opacity:0}}
.grpc4-a{animation:grpc4-left 12s ease-in-out infinite}
.grpc4-b{animation:grpc4-right 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.grpc4-a,.grpc4-b{animation:none;opacity:1;transform:translate(445px,0)}}
</style>
<text class="grpc4-label" x="380" y="28">One connection carries many RPC streams</text>
<rect class="grpc4-box" x="40" y="55" width="150" height="70" rx="10"/>
<text class="grpc4-label" x="115" y="84">client channel</text>
<text class="grpc4-small" x="115" y="107">one TCP socket</text>
<rect class="grpc4-box" x="315" y="55" width="150" height="70" rx="10"/>
<text class="grpc4-label" x="390" y="84">L4 balancer</text>
<text class="grpc4-small" x="390" y="107">choice at connect</text>
<rect class="grpc4-box" x="580" y="50" width="140" height="74" rx="10"/>
<text class="grpc4-label" x="650" y="79">backend A</text>
<text class="grpc4-small" x="650" y="102">selected</text>
<rect class="grpc4-box" x="580" y="183" width="140" height="74" rx="10"/>
<text class="grpc4-label" x="650" y="212">backend B</text>
<text class="grpc4-small" x="650" y="235">idle for this flow</text>
<path class="grpc4-line" d="M190 90 H315 M465 90 H580"/>
<path class="grpc4-line" d="M465 110 C520 110 530 220 580 220" stroke-dasharray="5 6" opacity=".45"/>
<circle class="grpc4-dot grpc4-a" cx="205" cy="90" r="8"/>
<circle class="grpc4-dot grpc4-a" cx="205" cy="90" r="8" style="animation-delay:1.5s"/>
<circle class="grpc4-dot grpc4-a" cx="205" cy="90" r="8" style="animation-delay:3s"/>
<circle class="grpc4-dot grpc4-b" cx="205" cy="90" r="8"/>
<text class="grpc4-small" x="380" y="157">RPC 1, RPC 2, RPC 3 share the same selected path</text>
<text class="grpc4-small" x="380" y="275">A new connection is needed for another L4 selection</text>
</svg>
<figcaption>New RPCs move through the same established connection and reach backend A; the L4 balancer does not choose again for each stream.</figcaption>
</figure>

The animation deliberately follows multiple RPC markers along the same connection. It does not claim that one gRPC channel always has one socket or that every new connection must select B. It shows the choice boundary that matters. A fresh connection gives an L4 balancer another opportunity to choose; an existing connection does not. A long-lived bidirectional stream is even more strongly attached because the RPC itself cannot move halfway through. That is why the gRPC performance guide warns that started streams cannot be load balanced again.

Here is an illustrative count. One client sends 1,000 equal-cost RPCs over one connection to a four-replica service. Under a connection-pinning L4 device, the backend chosen for that connection receives 1,000 RPCs from this client and the other three receive zero from it. The average across replicas is 250, but no replica actually gets 250 from this client. If the client instead holds four independent connections and a hypothetical perfect connection round robin assigns one to each backend, then 250 RPCs per connection yields 250 per backend. Real clients may send different amounts of work on each connection, and real L4 policies need not assign four connections perfectly. The arithmetic illustrates granularity, not a capacity prediction.

Even a large number of connections may not solve skew if work sizes differ. Suppose two connections each carry 100 RPCs, but one carries CPU-heavy requests costing an illustrative 50 ms each and the other carries cheap requests costing 2 ms each. Their backend demand is 5,000 ms versus 200 ms of CPU service time by multiplication. A connection-count dashboard reports an even 1:1 distribution. A request-count dashboard also reports 100:100. Only cost-weighted work or CPU per backend exposes the difference. This is why routing algorithms such as least-request and outlier detection exist above simple flow hashing, though their details belong in [balancing algorithms](/blog/software-development/networking/balancing-algorithms-round-robin-least-request-and-the-power-of-two-choices).

The [gRPC load balancing design](https://github.com/grpc/grpc/blob/master/doc/load-balancing.md), checked on 2026-09-30, describes `pick_first` as the default policy when service config does not specify another. It connects to the first reachable address and sends all RPCs on that selected address until connectivity changes. `round_robin` creates subchannels for addresses and chooses among ready ones per RPC. The distinction matters because client-side `round_robin` only helps when the resolver supplies the *backend addresses*. If it supplies a single VIP, rotating over one address is no distribution at all. Inspect what the client actually resolved and how its gRPC resolver/policy interprets those addresses before changing policy names.

An L7 proxy can terminate and parse the downstream HTTP/2 connection, then choose an upstream backend per RPC stream. That may be the right trust boundary or operational choice, but a proxy also has an upstream pool. Its pool policy, stream limits, endpoint discovery, and draining behavior can produce a second imbalance. A client-side policy can make per-call choices without a request-level proxy hop, but it must receive endpoint membership and carry the operational burden of health and policy configuration. The [gRPC team's 2017 load balancing discussion](https://grpc.io/blog/grpc-load-balancing/) sets out these L4, L7, and client-side choices. [Service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) treats the broader membership and ownership question.

![Comparison of L4 connection balancing, gRPC client policy, and L7 stream routing](/imgs/blogs/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap-5.webp)

| Placement | What it can select | What it must know | Cost or failure mode | Source |
| --- | --- | --- | --- | --- |
| L4 balancer | New transport connection | Transport tuple and backend pool | Many RPCs on one connection remain pinned | [gRPC load balancing, 2017](https://grpc.io/blog/grpc-load-balancing/) |
| gRPC client `round_robin` | RPC before it starts | Distinct backend addresses and ready subchannels | Requires correct resolver membership; started streams stay put | [gRPC LB design](https://github.com/grpc/grpc/blob/master/doc/load-balancing.md) |
| L7 HTTP/2 proxy | RPC stream after downstream termination | HTTP/2 and upstream endpoint state | Extra proxy work and its own pool or flow limits | [gRPC load balancing, 2017](https://grpc.io/blog/grpc-load-balancing/) |

The table is about capability, not a universal ranking. If a client opens thousands of similarly loaded short-lived connections, L4 may balance acceptably and keep proxy complexity low. If a small number of very hot gRPC channels carry most work, a connection-level choice is the wrong unit. If service boundaries require centralized authorization or route policy, an L7 hop can be justified for reasons beyond balancing. The network diagnosis tells you what granularity you currently have; the architecture decision follows.

## 8. Case file: Yik Yak's migration found two pooling layers

![Yik Yak 2017 gRPC migration path with Kubernetes Service and nghttp2 ingress reuse](/imgs/blogs/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap-6.webp)

Miguel Mendez's [Yik Yak engineering account, published April 12, 2017](https://grpc.io/blog/yikyak/), describes a migration to Google Cloud Platform using gRPC and Kubernetes. The post does not specify an exact day for each load test or deployment change, so April 12 is the publication date, not a claimed incident start. The observed problem was distribution: reusing an HTTP/2 connection meant requests did not spread across servers in the Kubernetes Service. At that time, the team chose to dial for each call so the Service would make a fresh connection-level selection. The author reports about 1 ms of added response time in their setup. That is a local result, not a safe default for every modern service.

The story did not end at the first balancing hop. In load tests after that workaround, the author reports the busiest gRPC server around 50% CPU and the least busy around 20% after several minutes of warmup. Their nghttp2 ingress preferred servers with already established upstream connections, which sustained uneven distribution. Removing that ingress reduced the variance in their graphs. The numbers and mechanism are from the [owner's published account](https://grpc.io/blog/yikyak/); they are not measurements from this site. The source does not establish that every nghttp2 deployment behaves this way today, nor does it compare modern client policies.

Separate the causal roles. The trigger for the first imbalance was an HTTP/2 connection reused across calls under a Service that selected at connection granularity. A workaround created fresh client-side connections, producing more selection opportunities. A second pooling layer in the ingress remained a contributing condition. The blast radius was visible as CPU skew across backend replicas even though the client appeared to be making many calls. The transferable guardrail is to draw and measure every connection pool between the caller and backend. Rebalancing one hop can leave the next hop's affinity untouched.

If this were our incident, the first evidence request would be a per-backend graph of active TCP connections beside per-backend RPC rate and CPU, with the same time axis. Then we would ask for the gRPC client's resolved address list and active policy. At the ingress, we would compare downstream connection count, downstream stream count, upstream connection count, and upstream stream count by backend. A mismatch between balanced downstream calls and skewed upstream work localizes the problem to the ingress. Without those four views, a blanket instruction to add replicas or restart clients is guesswork.

The case also shows why a request-level proxy must be evaluated at its *upstream* routing boundary, not just by whether it speaks HTTP/2 downstream. A proxy can parse a stream and still prefer an existing upstream connection or endpoint under its configured policy. The client may see one logical endpoint and never know that the proxy's backend choice is stale. This lesson composes with [connection pools and tail latency](/blog/software-development/networking/connection-pools-and-where-your-tail-latency-actually-lives) and with [service discovery](/blog/software-development/networking/service-discovery-from-dns-to-registries-to-xds), where membership staleness can affect the endpoint set presented to a balancer.

## 9. Diagnose at the selection boundary

![Decision tree for locating gRPC imbalance by comparing connection, stream, and RPC counts](/imgs/blogs/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap-7.webp)

When a backend is hot, measure the routing unit before changing it. Start with the client target and resolved addresses. If the client resolves only a Service VIP, a client-side policy cannot directly see individual replicas through that name. If it resolves replica addresses, inspect the active gRPC policy and ready subchannels. Next count established transport connections by backend. If the distribution is skewed there, the flow selection or connection population is skewed. If connections look even but RPCs do not, inspect stream assignment, long-lived calls, and per-connection workload. Finally compare upstream and downstream counts at each L7 proxy. The first hop where distribution changes is the place to intervene.

On Linux, `ss -tnp` identifies established sockets and their peer addresses on a host where you have permission to inspect the process. On a client with a connection to a VIP, `ss` may show only the VIP, not the chosen backend behind destination NAT. Inspect conntrack or balancer telemetry at the translating hop, or compare per-backend accepted sockets on the server side. Do not claim that a client-side socket list alone proves which replica received the flow. With TLS, a packet capture can count transport flows and bytes but cannot decode gRPC headers without endpoint keys or application-level tracing. Use method/status counters and proxy logs for stream-level evidence.

Useful counters have different denominators. `connections_opened` is a flow arrival count. `active_connections` is a snapshot. `rpc_started` counts calls. `rpc_active` counts work in flight. `messages_sent` can be many per streaming RPC. `bytes_sent` may be dominated by a few large calls. Keep the denominator in every chart label. If one backend has the same RPC count as another but much higher CPU, the balancing problem may be *work cost*, not request placement. Inspect method mix and payload size before blaming the network.

| Observation | Likely boundary to inspect | Discriminating evidence | First safe action | Source |
| --- | --- | --- | --- | --- |
| One backend owns most connections and RPCs | L4 flow selection or tiny client connection population | Server-side accepted sockets plus RPC starts | Confirm client target and policy; add per-call routing only if endpoint discovery supports it | [gRPC LB design](https://github.com/grpc/grpc/blob/master/doc/load-balancing.md) |
| Connections even, RPCs uneven | Streams per connection or long-lived RPCs | Active streams and RPC rate by connection/backend | Inspect channel and stream lifetime | [gRPC performance guide](https://grpc.io/docs/guides/performance/) |
| Downstream proxy RPCs even, upstream backend RPCs uneven | Proxy upstream pool | Proxy upstream requests and active connections by endpoint | Correct proxy endpoint policy or draining | [Yik Yak case, 2017](https://grpc.io/blog/yikyak/) |
| RPCs even, CPU uneven | Handler cost or method mix | CPU time, method, payload size, and in-flight work | Balance weighted work or reduce costly path | Derived diagnostic classification |
| Client waits before headers leave | Stream concurrency or local queue | Pending calls, ready subchannels, negotiated stream limit | Inspect saturation before enlarging pool | [gRPC performance guide](https://grpc.io/docs/guides/performance/) |

The table is a runbook entry, not a promise that one counter has one cause. It narrows the next measurement. A backend may be slow because its disk is unhealthy, because an L4 device pins a hot flow, or because a proxy retries selectively. The first boundary where the counts diverge is more useful than the last dashboard where latency rises. This is the same habit the [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) develops across the whole series.

When choosing a fix, name its second-order cost. Client `round_robin` needs distinct, current backend addresses and may increase connections. An L7 proxy parses and routes streams but adds a hop and needs its own upstream balancing, health, and draining policy. More channels can ease stream-slot queueing, but a large pool consumes sockets and may still map unevenly behind an L4 VIP. Short connection lifetimes force new L4 choices but add handshakes and can interrupt long-lived streams. Keepalive reduces idle reconnects but can preserve an unlucky affinity. Every control changes a measurable resource or failure mode.

### Make the client policy visible

The policy name in a configuration file is weak evidence until you know which target string received that configuration. The [gRPC service-config guide](https://grpc.io/docs/guides/service-config/) says config is scoped to a target, may be returned by name resolution or provided programmatically, and defaults to `pick_first` when no other policy is supplied. It shows `loadBalancingConfig` with `round_robin`, along with per-method timeout and retry settings. A deployment can therefore have an intended policy in one file and a different effective policy for the target the code dials. Log the exact target, resolver result, effective service config, ready subchannels, and the remote endpoint of each socket. The [official gRPC debugging guide](https://grpc.io/docs/guides/debugging/) describes Channelz and `grpcdebug` for inspecting channels, subchannels, sockets, RPC counts, and resolution state where supported.

Suppose an operator switches a Go client from `pick_first` to `round_robin` but keeps the target at a single Kubernetes Service VIP. The resolver may still present one address. One address gives a round robin policy no second endpoint to rotate to. The operator may see the policy string change but no backend distribution change. A headless Service or an endpoint-aware resolver can expose individual backend addresses, but then the client must receive timely membership updates and handle draining correctly. Do not change the DNS shape without reviewing health, security boundaries, and certificate identities. The [custom name-resolution guide](https://grpc.io/docs/guides/custom-name-resolution/) explains that resolver output includes addresses and can include service config; it does not make every ordinary DNS name an endpoint inventory.

Channelz is useful because it links the logical and transport scopes that dashboards often split. Start at the top channel for the client target, inspect child subchannels, and then inspect their sockets and remote addresses. The [gRPC team's Channelz walkthrough from September 5, 2018](https://grpc.io/blog/a-short-introduction-to-channelz/) shows a concrete case where one subchannel held the failed calls while sibling subchannels succeeded. The example is a diagnostic demonstration from that date, not proof that a particular production service has the same failure. Its method is general: walk channel to subchannel to socket, comparing counts at each step. That lets you ask whether an RPC even had a viable connection before chasing a server-side tail latency trace.

### Keep retry policy inside the caller's budget

Retries can hide transport races and can also multiply load on a hot backend. The [gRPC retry guide](https://grpc.io/docs/guides/retry/) distinguishes transparent retry, which the library may perform when it knows an attempt was not processed, from a configured retry policy with explicit attempts, backoff, and retryable codes. It says an RPC is committed for retry purposes when response headers arrive. That is a transport boundary, not a universal statement about whether an application's database write committed. It also says a default retry policy is not supplied even though certain transparent retries can occur. Count attempts separately from original calls when reconstructing load.

As a derived illustration, a caller makes 100 logical RPCs, each allowed up to three attempts. If all fail twice before succeeding, the backend sees 300 attempts, not 100. That is a 3× attempt multiplier under the stated worst-case pattern. If the same hot connection or endpoint is chosen for the attempts, retries intensify the existing skew. A per-call balancer may choose a different ready subchannel for a new attempt, but the exact retry placement depends on the policy and implementation. Do not assume retry equals failover. Record attempt number, selected backend, response headers, status, and deadline remaining. For writes, configure retries only where operation semantics and idempotency make repeated attempts safe.

The deadline constrains the entire logical call, including attempts and backoff. A child cannot spend the full parent budget on every retry and still honor the parent's deadline. Consider an illustrative 300 ms parent budget: the first attempt consumes 120 ms, the backoff consumes 60 ms, and the second attempt starts with at most 120 ms before response reserve. The arithmetic is 300 − 120 − 60 = 120 ms. If the service needs 200 ms to perform useful work under the current conditions, another attempt is unlikely to help. The exact backoff, timeout encoding, and cancellation timing are implementation choices; this budget calculation is an explanatory model for review. The [request hedging guide](https://grpc.io/docs/guides/request-hedging/) likewise states that the deadline applies to the whole set of hedged requests, not a fresh allowance per attempt.

### Drain a long-lived stream deliberately

The same connection reuse that makes gRPC efficient changes deploy behavior. A server can become unready for new work while active streams remain. Abruptly killing it can fail every in-flight stream on its connection; leaving it forever can prevent a rollout from completing. The [gRPC graceful-shutdown guide](https://grpc.io/docs/guides/server-graceful-stop/) describes stopping new RPCs while allowing in-flight calls a bounded interval to finish. At the HTTP/2 layer, `GOAWAY` helps peers stop creating new streams on a connection while already accepted streams proceed, as specified by [RFC 9113](https://www.rfc-editor.org/rfc/rfc9113). The deployment must still give the application time and a reconnection path.

For a unary method that completes quickly, a bounded drain can allow most calls to finish. For a stream intended to last hours, waiting for natural completion may be equivalent to never draining. Give the application a restartable stream protocol: resume token, sequence number, or idempotent subscription offset where its semantics allow one. Then define a maximum connection or stream age and communicate it before forceful termination. A keepalive PING cannot migrate a stream, and a client-side `round_robin` policy cannot relocate an RPC already in progress. Reconnection is a new RPC with its own correctness rules.

Review the order of operations during a deploy. Remove the endpoint from *new* selections, announce drain, give current RPCs a grace period, then stop the process. If the discovery update reaches the client after the server exits, that gap can produce failures even when the shutdown handler is correct. If an L7 proxy keeps an old upstream connection while its endpoint set changes, the same stale choice can persist at the proxy. Watch the active stream count, rejected new calls, GOAWAY events, connection age, and forced termination count on one time axis. This is the wire-level complement to [health checks and self-healing](/blog/software-development/microservices/health-checks-readiness-liveness-and-self-healing).

The senior rule is to change one selection boundary at a time. First prove that requests are pinned because the existing L4 hop only balances connections. Then decide whether exposing backend addresses to the client or adding an L7 routing hop fits the ownership model. Measure again at the next hop. If distribution improves but tail latency rises, check added proxy queueing, handshakes, and stream-slot limits before calling the first fix a failure. That sequence makes the intervention reviewable and reversible.

## Run it yourself

### Question

Can a connection-level balancer alternate *new connections* fairly while keeping all requests on one persistent connection pinned to one backend? This isolates the selection boundary responsible for the gRPC trap. The small lab deliberately uses a line protocol instead of gRPC, so it proves connection affinity rather than HTTP/2 frame syntax. The [HTTP/2 companion post](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer) and the protocol document above provide the stream mapping.

### Preconditions

Use the Linux `c` and `s` namespaces and `10.77.0.1/30` to `10.77.0.2/30` veth from [post 1's setup](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The experiment needs Python 3, `iproute2`, and permission to run commands inside those namespaces. On macOS, run that setup in a privileged Linux VM; `ip netns` is not a macOS facility. This lab opens only `10.77.0.2:8080`, `:18081`, and `:18082` inside namespace `s`; check those ports before starting. It does not change routes, qdiscs, packet filters, or sysctls. Run as a user with the required namespace privilege, normally root inside the lab VM.

```bash
set -euo pipefail
ip netns list | grep -E '^(c|s)( |$)'
ip -n c -br addr show c0
ip -n s -br addr show s0
ip -n c route get 10.77.0.2
ip netns exec c ping -c 1 -W 1 10.77.0.2
ip netns exec s ss -ltn '( sport = :8080 or sport = :18081 or sport = :18082 )'
python3 --version
ip netns exec c python3 --version
ip netns exec s python3 --version
```

The `ss` command should print only its header. If any of those ports already has a listener, stop here and choose a clean lab VM. The `ping` output should show one reply, proving address reachability, though it does not prove application health. The proxy below is a deliberately minimal L4 device: it chooses A or B when `accept()` returns and forwards bytes in both directions without parsing request lines.

### Baseline

Create the one-file lab and launch it inside `s`. The script prints each accepted connection's chosen backend to its log. The two backend handlers return their identifier on every request line. A real gRPC server would carry multiple RPC streams over HTTP/2 on a connection; the line protocol is the controlled stand-in for those multiple logical requests.

```bash
cat > /tmp/grpc-l4-affinity.py <<'PY'
import itertools
import select
import socket
import socketserver
import threading

HOST = "10.77.0.2"
BACKENDS = [(HOST, 18081, b"A"), (HOST, 18082, b"B")]
choices = itertools.count()

class Echo(socketserver.StreamRequestHandler):
    def handle(self):
        while True:
            line = self.rfile.readline()
            if not line:
                return
            self.wfile.write(self.server.backend_id + b"\n")
            self.wfile.flush()

class Server(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

def backend(port, name):
    srv = Server((HOST, port), Echo)
    srv.backend_id = name
    threading.Thread(target=srv.serve_forever, daemon=True).start()

def copy_both(client, upstream):
    with client, upstream:
        live = [client, upstream]
        while live:
            readable, _, _ = select.select(live, [], [])
            for src in readable:
                chunk = src.recv(65536)
                if not chunk:
                    return
                dst = upstream if src is client else client
                dst.sendall(chunk)

for address, port, name in BACKENDS:
    backend(port, name)
with socket.socket() as listener:
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind((HOST, 8080))
    listener.listen()
    print("ready", flush=True)
    while True:
        client, peer = listener.accept()
        address, port, name = BACKENDS[next(choices) % len(BACKENDS)]
        print("accepted", peer, "backend", name.decode(), flush=True)
        upstream = socket.create_connection((address, port))
        threading.Thread(target=copy_both, args=(client, upstream), daemon=True).start()
PY
ip netns exec s python3 -u /tmp/grpc-l4-affinity.py > /tmp/grpc-l4-affinity.log 2>&1 &
echo $! > /tmp/grpc-l4-affinity.pid
sleep 1
ip netns exec s ss -ltn '( sport = :8080 or sport = :18081 or sport = :18082 )'
```

Read: the `ss` output must show three LISTEN sockets in namespace `s`. The exact PIDs and socket buffers vary. If the script did not start, read `/tmp/grpc-l4-affinity.log` and stop rather than running the client against another service. Now send twenty request lines over *one* socket from namespace `c`.

```bash
ip netns exec c python3 - <<'PY'
from collections import Counter
import socket

counts = Counter()
with socket.create_connection(("10.77.0.2", 8080), timeout=3) as sock:
    stream = sock.makefile("rwb", buffering=0)
    for _ in range(20):
        stream.write(b"call\n")
        counts[stream.readline().strip().decode()] += 1
print("persistent_connection", dict(counts))
PY
cat /tmp/grpc-l4-affinity.log
```

Read: `persistent_connection` should show exactly `{'A': 20}` or `{'B': 20}`. The proxy log should show one `accepted` line for this baseline. The A/B label depends on which connection the proxy accepted first after startup; what matters is that there is one backend identifier for all twenty logical requests. Because this is a local namespace lab, it should complete within a few seconds on an idle VM. Scheduler load may change elapsed time but not the count if all calls complete.

### Apply one change

Change only connection reuse: send the same twenty lines as one line per fresh socket. Do not restart the proxy, change its selection rule, or alter the backend handlers.

```bash
ip netns exec c python3 - <<'PY'
from collections import Counter
import socket

counts = Counter()
for _ in range(20):
    with socket.create_connection(("10.77.0.2", 8080), timeout=3) as sock:
        stream = sock.makefile("rwb", buffering=0)
        stream.write(b"call\n")
        counts[stream.readline().strip().decode()] += 1
print("fresh_connections", dict(counts))
PY
```

### Compare

```bash
cat /tmp/grpc-l4-affinity.log
ip netns exec s ss -ltn '( sport = :8080 or sport = :18081 or sport = :18082 )'
```

Read: `fresh_connections` should show `{'A': 10, 'B': 10}` in either key order because this proxy deterministically alternates on each new accept. The log should now contain twenty-one `accepted` lines total: one baseline connection and twenty treatment connections. The three listening sockets should still be present. If you see timeouts or a different accepted count, the lab did not complete cleanly; check the log and namespace reachability. The exact wall-clock duration can vary with VM scheduling, but the request count and connection count are deterministic in this script.

The experiment shows the selection granularity. An L4 balancer can be fair when given many independent connections and still send every logical request on one persistent connection to a single backend. In actual gRPC, one HTTP/2 connection carries separate RPC streams instead of newline-delimited requests, but the L4 balancer's inability to choose again inside that connection is the same. Do not infer that production should dial for every RPC. The treatment is a diagnostic contrast, not a recommended client configuration.

### Reset

```bash
if [ -f /tmp/grpc-l4-affinity.pid ]; then
  kill "$(cat /tmp/grpc-l4-affinity.pid)"
  rm -f /tmp/grpc-l4-affinity.pid
fi
rm -f /tmp/grpc-l4-affinity.py /tmp/grpc-l4-affinity.log
ip netns exec s ss -ltn '( sport = :8080 or sport = :18081 or sport = :18082 )'
```

The final `ss` output should again contain only its header. Cleanup targets the one process and two files created by this lab; it does not delete the shared `c` or `s` namespaces. If you run a packet capture while extending the experiment, keep it in the lab and filter to these ports. Captures of real gRPC traffic may contain credentials or application data after decryption and should be handled accordingly. On a production host, a safer read-only first pass is `ss -tnp` plus per-backend connection and RPC counters at the relevant balancer or server; do not run this proxy or namespace setup on a production interface.

## Key takeaways

- A gRPC RPC is an HTTP/2 stream inside a connection. A message is length-prefixed inside DATA, and final gRPC status is carried in trailers.
- The client deadline is an end-to-end budget. Forward the remaining time, propagate cancellation, and remember that a timeout cannot roll back a committed side effect.
- Keepalive tests a connection, not method health. Negotiate PING behavior with servers and intermediaries before shortening intervals.
- Message-size limits, HTTP/2 frame size, flow-control windows, and concurrent-stream limits are separate constraints. Measure the failing boundary before tuning one of them.
- An L4 balancer chooses a backend for a transport connection. Per-RPC balance needs backend-aware client policy or an L7 hop that actually routes streams; every intermediate pool deserves inspection.

## Further reading

The [gRPC HTTP/2 protocol document](https://github.com/grpc/grpc/blob/master/doc/PROTOCOL-HTTP2.md) is the wire reference; [RFC 9113](https://www.rfc-editor.org/rfc/rfc9113) defines HTTP/2 stream, frame, flow-control, and PING behavior. The official [deadline](https://grpc.io/docs/guides/deadlines/), [cancellation](https://grpc.io/docs/guides/cancellation/), [keepalive](https://grpc.io/docs/guides/keepalive/), and [load-balancing](https://github.com/grpc/grpc/blob/master/doc/load-balancing.md) guides cover implementation policy. For a companion view of idle proxies and reconnect storms, read [WebSockets, SSE, and long-lived connections](/blog/software-development/networking/websockets-sse-and-long-lived-connections). For the broader diagnostic map, use [the senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model).
