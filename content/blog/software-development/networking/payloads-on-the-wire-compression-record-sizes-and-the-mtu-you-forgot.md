---
title: "Payloads on the Wire: Compression, TLS Records, and the MTU You Forgot"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Learn to separate serialization, compression, TLS record timing, and path MTU failures with wire-level measurements and a repeatable lab."
tags:
  [
    "networking",
    "distributed-systems",
    "compression",
    "gzip",
    "brotli",
    "zstd",
    "tls",
    "protobuf",
    "path-mtu",
    "performance",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot-1.webp"
---

A service call succeeds with a tiny response and stalls when the response has real data. Another call gets slower after you enable compression, although the response is one third the size. A third has a good median and an ugly time to first byte. All three can be described vaguely as a payload problem. They have different causes, and changing the wrong knob can make each one worse.

The request path from [what happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) ends with two easily confused clocks. **Time to first byte** stops when the client receives the start of the response. **Transfer time** ends when it has the response body. Compression can reduce transfer time while increasing the wait for the first byte. A path MTU failure can let the first byte arrive and then stop the transfer altogether. The opening latency ladder marks those two clocks without pretending that a sample number applies to every network.

![The request latency ladder highlights first byte and transfer as separate payload-sensitive phases.](/imgs/blogs/payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot-1.webp)

The diagram is the mental model: find the clock that moved, then ask which byte boundary moved it. A serialized object becomes a compressed representation, one or more TLS records, TCP segments or QUIC datagrams, and finally IP packets. Each boundary has its own size and buffering rules. The application can control some of them, but it does not get to assume that a write maps to a packet.

This post follows one response down that stack. We will calculate the CPU-for-bytes break-even, measure JSON and Protocol Buffers with the same logical data, distinguish a TLS record from a TCP segment, and diagnose the classic large-payload stall. The [networking capstone](/blog/software-development/networking/the-senior-engineers-network-mental-model) puts these measurements into the larger incident workflow. The [API performance post](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) owns product-level payload design; here we own what crosses the wire.

## 1. Four sizes, four different questions

When somebody says that a response is 100 KB, ask which layer counted it. A JSON encoder may produce 100 KB before HTTP compression. An HTTP server can gzip that into 28 KB of content-encoded bytes. TLS then adds record framing and authentication data. TCP breaks the stream into segments that fit its current maximum segment size, or MSS. IP puts headers around those segments. A packet capture can report a fifth quantity: link-layer frame bytes. None of these numbers is interchangeable.

The distinctions matter because a size reduction at one layer does not imply a proportional reduction at another. Small responses may gain a compression wrapper and get larger. A 16 KB TLS record can cross the network as many TCP segments. A `write()` call can be coalesced with the next write, split by the TLS library, or batched by the kernel. TCP segmentation offload can even make a capture on the sending host appear to contain packets larger than the physical path carries. At the receiving end, GRO can combine packets before a capture sees them. Check capture point and offload state before treating packet lengths as physical frames.

| Size | Counts | Does not count | Useful instrument | Source |
| --- | --- | --- | --- | --- |
| Logical object | Values and fields after parsing | JSON punctuation or protobuf tags | Application instrumentation | Defined here |
| Serialized body | JSON or protobuf bytes before HTTP content coding | TLS and IP overhead | `wc -c` on encoder output | Defined here |
| Content-encoded body | Compressed representation bytes | TLS and IP overhead | `curl --raw` plus response headers | [RFC 9110, June 2022](https://www.rfc-editor.org/rfc/rfc9110.html#section-8.4) |
| TLS record | Encrypted fragment plus TLS framing and authentication overhead | A fixed number of IP packets | TLS endpoint instrumentation or controlled capture | [RFC 8446, August 2018](https://www.rfc-editor.org/rfc/rfc8446.html#section-5.2) |
| IP packet | IP header and IP payload | Ethernet framing | `tcpdump` at a known capture point | [RFC 4821, March 2007](https://www.rfc-editor.org/rfc/rfc4821.html#section-3) |

`Content-Encoding` describes a transformation of the HTTP representation. It is distinct from HTTP/1.1 `Transfer-Encoding: chunked`, which frames a message whose length is not known up front. The distinction in [RFC 9110, June 2022](https://www.rfc-editor.org/rfc/rfc9110.html#section-8.4) and [RFC 9112, June 2022](https://www.rfc-editor.org/rfc/rfc9112.html#section-7) prevents a common debugging mistake: turning off chunked framing does not turn off gzip, and `Content-Length` on a compressed response describes coded bytes, not the size of the JSON after decoding.

The practical measurement starts with three values: body bytes before compression, body bytes as transferred, and the times for first byte and complete transfer. Keep the same route, response data, cache state, connection reuse, and concurrency when comparing them. A browser DevTools "size" column may include caching or decoded-size presentation. Put the exact field name and capture method beside any benchmark result.

### The boundary rule

> A byte saved at the serializer is useful only if it survives content coding, record framing, and the path it actually takes.

That does not mean every optimization needs packet analysis. It means we should measure where the claimed win is supposed to appear. A protobuf migration that cuts serialized bytes but leaves compressed transfer bytes almost unchanged has a different business case from one that removes an entire RTT of transfer. Likewise, a `curl` response that hangs only above a threshold deserves a packet and PMTU check before another JSON refactor.

## 2. Compression is a CPU-for-bytes exchange

Compression trades compute and buffering for fewer bytes. The client may pay decompression CPU too, and the server may pay memory to keep a working dictionary. A cacheable asset compressed once at build time has a very different economics from a small personalized response compressed for every request. The algorithm name alone cannot select the right configuration.

HTTP content negotiation supplies the contract. A client advertises acceptable codings with `Accept-Encoding`; the server marks the chosen representation with `Content-Encoding`. A server cannot assume support just because a browser or another internal service supported an algorithm in a different request. `Vary: Accept-Encoding` matters when a shared cache stores variants. [RFC 9110, June 2022](https://www.rfc-editor.org/rfc/rfc9110.html#section-12.5.3) spells out negotiation and the meaning of `identity`, including the case in which a coding is unacceptable.

Gzip is the long-lived compatibility baseline. Brotli is a content coding defined by [RFC 7932, July 2016](https://www.rfc-editor.org/rfc/rfc7932.html). Zstandard has an HTTP content-coding registration in [RFC 8878, February 2021](https://www.rfc-editor.org/rfc/rfc8878.html). These documents define formats, not a universal performance ranking. The ranking depends on content, level, implementation, machine, dictionary, and whether we measure compression, decompression, or end-to-end completion. `br` or `zstd` must also be negotiated with the actual client population.

### A break-even model you can challenge

Here is an **explanatory model**, not an equation specified by an RFC. Let $B_0$ be uncompressed bytes, $B_c$ compressed bytes, $R$ available *goodput* in bits per second, and $T_c$ the server-side compression time added on the request's critical path. Assume transfer is the bottleneck, both representations start sending after they are fully ready, and decompression is either negligible or included in the client-side term $T_d$. Then:

$$
T_{\text{plain}} \approx \frac{8B_0}{R}, \qquad
T_{\text{coded}} \approx T_c + \frac{8B_c}{R} + T_d.
$$

Under those assumptions, compression improves completion time when:

$$
T_c + T_d \lt \frac{8(B_0-B_c)}{R}.
$$

The right-hand side is the time saved by transmitting fewer bits. It is a budget, not a benchmark. For a **derived example**, take a 100 KiB body that becomes 30 KiB. The saving is 70 KiB, or $70\times1024\times8=573{,}440$ bits. At 10 Mbit/s of actual goodput, that is $573{,}440/10{,}000{,}000=0.057344$ seconds, about 57.3 ms. If compression plus decompression adds 4 ms to the critical path, the model predicts an approximately 53 ms completion benefit. At 1 Gbit/s, the same bytes save only about 0.573 ms, so a 4 ms CPU cost would lose. These are derived scenarios, not claims about gzip, brotli, or zstd on a particular CPU.

![A conceptual CPU-versus-wire-time frontier places a compression choice on the break-even line.](/imgs/blogs/payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot-2.webp)

The model needs revision for streaming compression. If the compressor emits chunks while the application is still producing the body, CPU and transmission overlap. Then the critical path is closer to the maximum of production and transfer stages than to their simple sum, with queueing and flush behavior included. It also changes under congestion, loss, HTTP flow control, and cellular radio scheduling. The useful move is not to worship the formula. It is to measure each term and decide whether the saved bits are expensive on *this* path.

### When each coding belongs in the conversation

For static JavaScript, CSS, and text assets, compress once during a build and serve the variant the client negotiated. A slower build-time compressor may be acceptable because request-time CPU is near zero. For dynamic responses, first benchmark the levels actually available in the server library. An algorithm's maximum compression level may reduce bytes while extending the critical path and consuming the CPU capacity that keeps queueing low. For small messages, skip compression if the wrapper and setup cost dominate or if the body contains little redundancy. For already compressed JPEG, AVIF, or archive data, extra HTTP compression often buys little and can cost CPU. Those are hypotheses to test on your payload mix, not fixed byte thresholds.

Measure throughput and p99 at the same offered load, not merely the compression ratio of one file. If CPU utilization rises enough to create a queue, the server's time to first byte and the p99 can regress even when every response has fewer bytes. This connects to [queueing and bufferbloat](/blog/software-development/networking/bufferbloat-queueing-and-the-latency-you-added-yourself): the bottleneck can migrate from the link to the compressor worker pool. Track compressed bytes, CPU per response, queue wait, and transfer duration together.

The break-even model can be rearranged to expose a **goodput threshold**. If added critical-path CPU cost is $T_c+T_d$, compression improves the modeled completion time only when $R \lt 8(B_0-B_c)/(T_c+T_d)$. Using the derived 70 KiB saving above and an assumed 4 ms total CPU cost gives $R \lt 573{,}440/0.004=143{,}360{,}000$ bit/s, or about 143 Mbit/s. Below that goodput, the model favors compression; above it, the CPU term dominates. This threshold is not a deployment setting. It changes if the compressor streams, the client decompresses while receiving, or the service queues under load. Its value is that it tells you which measurements could overturn the decision.

The number of bytes saved is also response-specific. If the same 100 KiB body compresses only to 95 KiB, the saved bits are $5\times1024\times8=40{,}960$. At 10 Mbit/s, the budget is 4.096 ms, nearly the assumed CPU cost. A small change in CPU scheduling or payload content can reverse the result. This is why an algorithm comparison must include representative content and p99. A median compression ratio over highly repetitive test data can conceal the tails that users notice.

There is also a security boundary. If secret material and attacker-controlled text share a compressed context, compressed lengths can reveal information under suitable attack conditions. Do not enable response compression mechanically on sensitive reflected content. The specific risk depends on threat model, context reuse, and whether an attacker can observe sizes. Keep the content-coding decision in the same review as caching and response construction.

### The break-even changes when the server is busy

The arithmetic above prices one isolated response. A production service has many simultaneous responses. The more expensive compression mode can consume CPU that would otherwise serve the next request. Once utilization approaches the available worker capacity, queue wait can dominate a small byte saving. This is a reason to compare two load tests at the same offered request rate and data mix. Record CPU time spent compressing, runnable queue or worker-pool wait, p50 and p99 first-byte time, and p50 and p99 completion time. A single-file `time gzip` benchmark answers the codec's local cost; it does not answer what a saturated service will do.

Consider a **derived capacity example**. Suppose one server has 8 effective CPU cores available to its handler and compression work. At the target load, all noncompression work consumes 5 core-seconds per wall-clock second. A coding mode consumes 1 additional core-second per second, bringing demand to 6. Another mode consumes 3, bringing it to 8. Both might reduce the same 100 KiB response, but the second has no headroom under that assumed load. A small burst can create a queue even if the per-request compression microbenchmark seems acceptable. These numbers are deliberately chosen for arithmetic, not measured service capacity. The lesson is to price CPU at the target concurrency as well as milliseconds on one request.

Cache placement further changes the equation. A CDN can store a precompressed `br` variant and pay compression CPU once per asset version. An origin that varies a response by user may pay per request. A proxy that re-compresses an already compressed representation can waste CPU and sometimes enlarge the body. Check whether the cache key includes `Accept-Encoding` and whether the response's `Vary` header communicates that dimension. Inspect the actual client-visible `Content-Encoding`, because the origin's header may not be the last transformation. For a personalized API, it can be better to reduce unnecessary fields before choosing a more expensive codec. Bytes never serialized need neither compression CPU nor decompression CPU.

The latency model also depends on who pays. A mobile client on a weak CPU may be limited by decompression or JSON parse time after network transfer. A high-bandwidth datacenter client may see almost no transfer gain. A batch job may care about total bytes and billable egress more than first-byte latency. State the objective before tuning the level. If the objective is p99 user-visible completion, then measure that objective. If the objective is network capacity, report goodput and byte volume. The same compression setting can be good for one and poor for the other.

### Measure codec work without pretending it is the request

A useful microbenchmark records uncompressed bytes, compressed bytes, compression wall time, decompression wall time, CPU time, and peak allocation for a corpus of real samples. Warm up the runtime, include small and large bodies, and separate highly repetitive records from already compressed blobs. Run at least several iterations and report the median and distribution, not only the fastest sample. Pin the implementation and level. Then run the full HTTP path with the same body corpus and a fixed arrival rate. The first test explains *why* an algorithm behaves as it does; the second establishes whether it improves the service.

When interpreting a benchmark, distinguish savings per response from savings per second. If 10 percent of requests are large and compressible while the rest are tiny, an average ratio can hide which requests consume CPU. A conditional policy based on content type and body size may be appropriate, but it should be driven by measured distributions and negotiation support. Avoid a brittle single threshold copied from another deployment. Encoding a 200-byte error response at an extreme level has a different cost profile from serving a precompressed 2 MiB asset.

## 3. First byte is a flush decision as well as a bandwidth decision

A response can be 10 MB and still have a fast first byte if the server can emit an early useful chunk. A response can be 1 KB and have a slow first byte if an application buffer, compressor, proxy, or TLS stack waits before releasing it. Start by distinguishing production time from transmission time: `curl -w '%{time_starttransfer} %{time_total}'` measures client-visible timing, but it cannot by itself tell which component held the bytes.

TLS 1.3 permits a plaintext fragment of at most $2^{14}=16{,}384$ bytes per record, as specified in [RFC 8446, August 2018](https://www.rfc-editor.org/rfc/rfc8446.html#section-5.1). That is a maximum, not a required size and not the Ethernet MTU. [RFC 8449, August 2018](https://www.rfc-editor.org/rfc/rfc8449.html) defines a negotiated record-size-limit extension for endpoints that need smaller records. Implementations may make their own scheduling choices below the maximum. A server can flush a small record early or wait to fill a larger one. The record also carries framing and encryption overhead, so making every record tiny is not free.

![A packet timeline contrasts an early small TLS record with a buffered larger record.](/imgs/blogs/payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot-3.webp)

Imagine an application rendering a page in pieces. At time $t_0$ it has a 1 KiB heading and at $t_1$ it has the rest. If the whole response is buffered until $t_1$, the earliest possible first byte is after $t_1$ plus network travel. If the first piece is flushed at $t_0$, the client can see it earlier. That statement is a **causal model**. It does not assert that every TLS library flushes at a particular byte count or that a 1 KiB record is always faster. Sending an early record helps only when the application and every intermediary actually flushes it and when the client can use it.

Compression adds another buffer. A streaming gzip encoder can accept the first 1 KiB without emitting a useful coded chunk until flush. A proxy may buffer the response even after the application flushes. HTTP/2 and HTTP/3 add stream and connection scheduling. The [HTTP/2 flow-control post](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer) explains why a stream can be ready yet unable to advance. A packet capture at the origin may show an early record while the client sees it late because the proxy held or re-encrypted it. Capture at both sides of the suspected boundary.

The diagnostic question is precise: *when did the first application byte exist, when did it leave the TLS endpoint, and when did the client receive it?* Record an application timestamp, an egress capture or proxy access-log phase, and the client `time_starttransfer`. A large `time_starttransfer` with a small `time_total - time_starttransfer` says the wait occurred before or on the first byte's path. A small `time_starttransfer` with a large transfer tail points toward body production, flow control, bandwidth, retransmission, or MTU.

### TLS record size and packet size are different knobs

A 16 KiB TLS record is an encrypted application unit. It can be split over roughly a dozen ordinary TCP segments on a 1500-byte IPv4 path, depending on options, headers, and record overhead. A TCP segment contains a slice of a byte stream. A retransmission may cut that stream differently. A packet capture of TLS-over-TCP cannot safely infer application writes from segment boundaries alone. Even when a capture tool reconstructs TLS records, it cannot read encrypted application data without keys. The legitimate way to determine when the app produced bytes is instrumentation at the endpoint.

The cost of tiny records includes a larger fraction of framing and authentication overhead and more per-record work. The cost of huge buffering is delayed first byte and sometimes unfair scheduling among concurrent streams. Therefore set record and flush policy around the product goal: first renderable content, streaming event latency, or bulk throughput. Test with the actual client and intermediaries. Do not turn the 16 KiB upper bound into an application flush target by accident.

### A three-clock experiment for first-byte regressions

If a streaming endpoint is slow, put three timestamps on one request ID: when the application finishes its first useful chunk, when the TLS terminator sends the first encrypted application record, and when the client receives its first response byte. The first gap is application, compressor, proxy, or TLS buffering. The second gap is network, downstream proxying, or client scheduling. If TLS terminates at an edge, add a timestamp on both sides of that edge. This is a measurement plan, not a claim that any one implementation exposes all timestamps automatically.

`curl` supplies the client clock. Its `time_starttransfer` is measured from the beginning of the transfer and includes name resolution, connection establishment, TLS, request transmission, and server response wait. Compare it with `time_connect` and `time_appconnect` on a cold HTTPS connection; on a reused connection those setup phases are absent. If you want to isolate application response delay, subtract relevant earlier phases and state the caveats. Do not call `time_starttransfer` pure server time. `time_total` includes the transfer tail. For a repeated test, keep `curl` connection behavior fixed or the setup difference can masquerade as a record-size change. The [TLS handshake post](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs) locates those earlier phases on the same ladder.

Here is an exact read-only client probe for an endpoint you control. The empty output body is discarded, so the command records timing without writing response data. The `size_download` field reports bytes according to curl's transfer accounting; inspect the response headers to know whether content coding was used. This probe does not reveal TLS record boundaries on its own.

```bash
curl --silent --show-error --output /dev/null \
  --write-out 'connect=%{time_connect} tls=%{time_appconnect} first=%{time_starttransfer} total=%{time_total} size=%{size_download}\n' \
  'https://your-controlled-endpoint.example/stream'
```

Repeat it with a response whose first chunk is flushed early and one that is deliberately buffered, changing only flush policy. Put request IDs in server logs so each client timing line can be matched to a server-side production timestamp. If the first byte moves but body size does not, the evidence points toward the flush path. If the application says it flushed early but the client does not see it, instrument each proxy and the TLS terminator in order. Do not infer from the code calling `Flush` that all intervening layers complied.

## 4. JSON versus protobuf must be measured after every coding

The phrase "protobuf is smaller" usually refers to the serialized representation before HTTP compression. Protocol Buffers encode fields as numbered tags with wire types and use variable-width integers for many numeric values. The official [Protocol Buffers wire-format guide](https://protobuf.dev/programming-guides/encoding/) explains the tag, varint, length-delimited field, and repeated-field encoding. JSON instead carries textual field names, separators, quoted strings, and decimal spellings of numbers. Those properties suggest where protobuf may save bytes. They do not specify the ratio for your messages.

The comparison needs four cells: JSON and protobuf before compression, then the exact same logical data under the same negotiated content coding. Repeated JSON keys compress well. Short messages may be dominated by envelopes and transport overhead. Protobuf strings remain strings; an image stored as a `bytes` field does not magically shrink. Packed repeated numeric fields can be efficient, but small field tags and varints have their own overhead. Changing the shape or semantics of the object while changing the serializer invalidates the measurement.

![A comparison matrix separates serializer bytes from compressed wire bytes for identical data.](/imgs/blogs/payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot-4.webp)

Here is one real, deliberately narrow result. I ran the commands below on September 30, 2026, on an Apple M4 with macOS Darwin 25.3.0, Python 3.14.5, Apple gzip 475, brotli 1.2.0, zstd 1.5.7, and protoc 33.1. The fixture has 1,000 events with identical string fields and integer IDs from 0 through 999. The JSON encoder uses compact separators; protobuf uses the schema shown in the lab. These are **body byte counts**, measured with `wc -c`, not latency or a representative workload benchmark. That unusual repetition is useful because it exposes the difference between raw and compressed rankings. It must not be generalized to an arbitrary API.

| Same logical fixture | Raw body bytes | Gzip body bytes | Brotli body bytes | Zstd body bytes | Source |
| --- | ---: | ---: | ---: | ---: | --- |
| JSON | 105,902 | 2,962 | 1,382 | 1,401 | Local reproducible lab, September 30, 2026; commands below |
| Protobuf | 58,870 | 2,387 | 1,667 | 1,502 | Same lab and fixture; `protoc --encode=Batch` |

The raw protobuf message is $105{,}902-58{,}870=47{,}032$ bytes smaller, a derived reduction of about 44.4 percent relative to this JSON file. Under the tested gzip level, the difference is only $2{,}962-2{,}387=575$ bytes. Under the tested brotli level, protobuf is **285 bytes larger** than JSON; under the tested zstd level it is **101 bytes larger**. That is not a bug in protobuf and not proof that JSON is generally superior. The repeated keys and strings in this synthetic JSON are highly compressible, while the protobuf representation starts smaller and has a different redundancy pattern. The only defensible conclusion is about this fixture, these codec options, and these measured body files.

Notice the table says *body bytes*. It excludes HTTP headers, TLS record overhead, and IP headers. To claim a wire reduction, also measure transferred bytes at the HTTP client or packet boundary and name which one. A `curl` display can silently decode a content-encoded body. Use `--raw` when you need the coded bytes as delivered by the HTTP layer and save the response headers with the body. If an API gateway decompresses and recompresses, the origin's body size is not the client's body size.

If you have a JSON endpoint and a protobuf endpoint, compare responses under matching conditions: same data and field presence, same cache state, same content coding, same HTTP version, same connection warmth, and same concurrency. Take many samples of varied real payloads. Report median and tail sizes, not one celebratory record. The [gRPC and protobuf API design post](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming) covers schema evolution and contract ergonomics. A wire-size win is only one input to that architectural choice.

### Headers and envelopes can erase the apparent win

An RPC may carry a length prefix, metadata, tracing headers, authentication tokens, and an HTTP/2 frame envelope. If the application message is 90 bytes, saving 20 message bytes may barely move total request bytes. If the request repeats a large JSON structure thousands of times, the saving can be material. Measure both the message and end-to-end transfer to avoid confusing local codec efficiency with network efficiency.

HPACK or QPACK compresses HTTP field sections. It does not compress the JSON or protobuf body. Conversely, gzip on the body does not remove per-request headers. That boundary matters when an engineer reports "compression is on" without saying which compression. The [HTTP/2 post](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer) owns header and stream mechanics; we are measuring the representation after the serializer.

## 5. MTU is the smallest link, not the size of your response

Maximum transmission unit, or MTU, is the largest IP packet a link can carry in one piece under the definition used by the PMTUD RFCs. The path MTU is the minimum link MTU across the path, not the sender's local interface value. [RFC 4821, March 2007](https://www.rfc-editor.org/rfc/rfc4821.html#section-3) defines these terms. Tunnels add headers, so a path that begins on a 1500-byte Ethernet segment can include a smaller effective IP MTU later. A jumbo-enabled host does not make the whole path jumbo-capable.

For a simple **derived IPv4 example with no IP or TCP options**, a 1500-byte IP MTU leaves $1500-20-20=1460$ bytes of TCP data per segment. If a tunnel leaves a 1400-byte path MTU, the corresponding simple bound is $1400-20-20=1360$ bytes. TCP options, IPv6 headers, encapsulation, and implementation details change actual overhead, so treat those figures as accounting examples, not universal MSS settings. The peer's MSS advertisement says what it can receive at its endpoint; it is not proof that every intervening link can carry a segment of that size. [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html#section-3.7.1) specifies the MSS option's role, and [RFC 6691, July 2012](https://www.rfc-editor.org/rfc/rfc6691.html) clarifies TCP options and MSS sizing.

We can price segmentation without pretending that all segments turn into distinct physical frames. Take a 100 KiB TCP byte stream and the simple no-options IPv4 bounds above. At 1460 bytes per segment, $\lceil102{,}400/1460\rceil=71$ segments are needed. At 1360 bytes, $\lceil102{,}400/1360\rceil=76$. Five extra segments in this derived example mean more packet headers and packet-processing work, but they may be necessary for a 1400-byte path. They are far cheaper than repeatedly losing an oversized segment and timing out. Real segment counts differ with TCP options, TLS overhead, retransmission, offload, and exact body boundaries. Count actual wire packets at a suitable capture point before turning this simple bound into a throughput estimate.

For IPv6, the fixed base header is 40 bytes instead of IPv4's typical 20-byte no-options header, and extension headers or tunnels can add more. Do not use the IPv4 1360 calculation as an IPv6 clamp. The path MTU must include the packet as it exists at the narrow hop. If a tunnel adds its own outer IP header, the effective inner packet budget shrinks. The useful operational question is not "what is the NIC MTU?" but "how large is the final encapsulated IP packet at the smallest link?" That is why tunnel endpoints and middleboxes need to be part of the capture plan.

Classical path MTU discovery sends a packet that must not be fragmented. A router that cannot forward it should return an ICMP "Fragmentation Needed" message for IPv4 or Packet Too Big for IPv6. The sender lowers its packet size. If the feedback never reaches the sender, the handshake and tiny requests can work, then data stalls as full-size packets repeatedly disappear. [RFC 2923, September 2000](https://www.rfc-editor.org/rfc/rfc2923.html#section-2.1) documented the black-hole failure, including the role of firewalls that suppress ICMP and the danger of an eventual timeout.

![A before-and-after path shows a too-large packet and missing ICMP feedback, then successful smaller packets.](/imgs/blogs/payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot-5.webp)

MSS clamping is a mitigation at an edge that rewrites the MSS option in a TCP SYN so peers send smaller segments. If the tunnel's safe effective path MTU is known and the clamp accounts for its headers, that can keep TCP below the bottleneck. It is a TCP-specific configuration on the SYN path, not a repair for every UDP application or a substitute for working PMTUD. It can also waste capacity if clamped too low, and it must be applied in the right direction. Use a SYN capture to verify the offered MSS and a data capture to verify actual segment sizes. The setup is environment-specific; never apply a blanket `iptables` or `nft` command to a production interface from an article.

Packetization Layer PMTUD, described in [RFC 4821, March 2007](https://www.rfc-editor.org/rfc/rfc4821.html), uses transport-level probes and acknowledgments to discover a usable packet size even if ICMP feedback is absent. It can complement classical PMTUD and recover from black holes. This is not permission to drop ICMP deliberately. Correct delivery of authenticated or validated PMTU feedback avoids unnecessary probe failures and long recovery. Current Linux exposes `net.ipv4.tcp_mtu_probing` and related settings in its [kernel IP sysctl documentation](https://docs.kernel.org/networking/ip-sysctl.html). Inspect the current kernel and configuration before proposing a host-wide change.

### Jumbo frames do not rescue an undersized path

Jumbo frames can reduce packet-processing overhead on a controlled, consistently configured fabric. They require end-to-end support on that fabric, including switches, virtual interfaces, tunnels, and peers. A host reporting an MTU around 9000 on one NIC says nothing about the public internet path. If one hop is smaller and feedback fails, an oversized packet is still oversized. An isolated jumbo change can make the black-hole threshold more visible, not less. The safest test compares `ip link show`, `ip route get`, path-aware probes where permitted, and captures at the ingress and egress of the tunnel or router.

The most misleading symptom is selectivity. A `ping` default payload succeeds. A TLS handshake succeeds. A GET of a small health page succeeds. A larger API response times out. That pattern can be PMTU, but it can also be a proxy body limit, an HTTP/2 flow-control stall, an application deadlock, or packet loss. Look for repeated retransmissions of the same large sequence range and absent PMTU feedback. Use counters and captures to discriminate; do not diagnose by payload threshold alone.

### Why a response-size threshold can move after compression

Suppose a response serializes to 64 KiB and a compressor reduces it to 12 KiB. If the path drops only full-size packets, the compressed response may cross because its last packet is small or because a particular sender's segmentation behavior changes. A later release adds a field, changes compression ratio, or switches content coding, and the failure returns. The first apparent fix was only a shift in which packet sizes were emitted. This is another **explanatory scenario**, not a guaranteed outcome: even a 12 KiB response commonly contains multiple full-size packets, so compression may leave the black hole unchanged. You need a capture of actual segment sizes to know.

That caveat is crucial because otherwise a team may credit its gzip rollout with solving a network fault. A path-MTU black hole depends on IP packet size and the feedback path. The HTTP representation length affects how many packets are attempted, but it does not set each packet's size directly. TCP's current MSS, PMTU estimate, offload behavior, and congestion state participate. A tiny request succeeding tells you only that *those* packets crossed. It does not prove that the next full-sized segment will cross.

For a focused capture, start before the test, filter by one server IP and port, and stop promptly. Capture at the client and, if possible, the sender side of the narrow hop. On Linux, `tcpdump -ni c0 -s 160 -w /tmp/payload-mtu.pcap 'host 10.77.0.2 and (tcp port 8080 or icmp)'` is appropriate only when `c0` is the controlled `netlab` interface and the traffic is synthetic. The 160-byte snap length is enough for many IP/TCP-header questions but not every encapsulation or ICMP quote, so enlarge it if the missing field is truncated. Do not treat the capture as proof of on-wire frame size until checking offload at that capture point. Stop and secure the pcap, because even a small snap length can retain sensitive headers on real traffic.

Read packet timestamps, sequence ranges, IP lengths, the DF flag where applicable, and ICMP type/code or IPv6 Packet Too Big. A repeated large sequence range with increasing retransmission intervals, no advancement of the ACK, and no usable PMTU feedback is consistent with a black hole. It is not conclusive if the capture point could miss feedback or the loss is ordinary congestion. A capture on the narrow hop that sees the too-large packet being discarded and the feedback generated, paired with a sender capture that never sees the feedback, is much stronger. Cloudflare's case shows why the final hop to the owning server matters.

### The safe order of mitigations

First, correct the path so legitimate PMTU feedback reaches the sender. This can require firewall policy, ICMP handling through NAT, ECMP steering, or a tunnel configuration fix. Verify both directions and the actual flow owner. Next, enable or verify packetization-layer probing where the transport supports it, so a missing control message does not cause an indefinite stall. Third, consider scoped MSS clamping when the bottleneck is known and TCP crosses a tunnel or edge that cannot otherwise be corrected quickly. Last, assess jumbo settings only on a fully controlled fabric, with every hop and encapsulation included.

Each mitigation has a diagnostic cost. A clamp can make the incident disappear while hiding the broken feedback path; a future UDP workload can then rediscover it. A universal low MTU wastes packet-processing capacity on unaffected paths. An unvalidated host-wide sysctl can change behavior for unrelated applications. Make the smallest scoped change that the capture supports, and keep a rollback. The right end state is a path whose packet sizing remains correct after route changes, not a single payload size that happened to pass today's test.

## 6. An animation of a black hole

The key PMTUD state change is temporal. The sender transmits a packet sized for its local interface. A narrower hop discards it. The router's feedback does not reach the sender, so the sender retransmits at the same unusable size. A smaller probe or a correctly delivered ICMP message breaks the loop. The animation below makes the repeated failure and later success visible. Its sizes are the derived IPv4 example above, not a report of Cloudflare's production packet sizes.

<figure class="blog-anim">
<svg viewBox="0 0 800 300" role="img" aria-label="A 1500-byte IP packet reaches a 1400-byte tunnel and is dropped with feedback missing; a later 1400-byte packet crosses" style="width:100%;height:auto;max-width:800px">
<style>
.pmtu30-card{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.pmtu30-label{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}.pmtu30-note{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}.pmtu30-track{stroke:var(--border,#d1d5db);stroke-width:3}.pmtu30-big{fill:#ffc9c9;stroke:#b42318;stroke-width:2}.pmtu30-small{fill:#b2f2bb;stroke:#176b3a;stroke-width:2}.pmtu30-x{font:700 28px ui-sans-serif,system-ui;fill:#b42318;text-anchor:middle}.pmtu30-status{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
@keyframes pmtu30-large-flight{0%,5%{transform:translateX(0);opacity:1}33%,45%{transform:translateX(276px);opacity:1}53%,100%{transform:translateX(276px);opacity:0}}
@keyframes pmtu30-small-flight{0%,48%{transform:translateX(0);opacity:0}52%{transform:translateX(0);opacity:1}88%,100%{transform:translateX(555px);opacity:1}}
@keyframes pmtu30-drop{0%,29%{opacity:0}33%,54%{opacity:1}58%,100%{opacity:0}}
.pmtu30-bigmove{animation:pmtu30-large-flight 12s ease-in-out infinite}.pmtu30-smallmove{animation:pmtu30-small-flight 12s ease-in-out infinite}.pmtu30-dropmark{animation:pmtu30-drop 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.pmtu30-bigmove,.pmtu30-smallmove,.pmtu30-dropmark{animation:none}.pmtu30-bigmove{transform:translateX(276px);opacity:1}.pmtu30-smallmove{transform:translateX(555px);opacity:1}.pmtu30-dropmark{opacity:1}}
</style>
<rect class="pmtu30-card" x="20" y="24" width="180" height="60" rx="8"/><rect class="pmtu30-card" x="310" y="24" width="180" height="60" rx="8"/><rect class="pmtu30-card" x="600" y="24" width="180" height="60" rx="8"/>
<text class="pmtu30-label" x="110" y="60">sender, MTU 1500</text><text class="pmtu30-label" x="400" y="60">tunnel, MTU 1400</text><text class="pmtu30-label" x="690" y="60">receiver</text>
<line class="pmtu30-track" x1="110" y1="135" x2="690" y2="135"/><line class="pmtu30-track" x1="110" y1="230" x2="690" y2="230"/>
<rect class="pmtu30-big pmtu30-bigmove" x="102" y="111" width="88" height="42" rx="7"/><text class="pmtu30-label pmtu30-bigmove" x="146" y="137">1500 B</text><text class="pmtu30-x pmtu30-dropmark" x="400" y="146">×</text>
<rect class="pmtu30-small pmtu30-smallmove" x="102" y="206" width="88" height="42" rx="7"/><text class="pmtu30-label pmtu30-smallmove" x="146" y="232">1400 B</text>
<text class="pmtu30-status" x="400" y="184">ICMP feedback missing</text><text class="pmtu30-note" x="400" y="280">smaller probe or corrected MSS crosses the same path</text>
</svg>
<figcaption>The oversized IP packet disappears at the narrower tunnel while feedback fails to reach the sender; a later smaller packet can cross.</figcaption>
</figure>

The moving packet is an IP packet, not a TLS record or a JSON message. That distinction matters: compression can change how many packets are needed, but it cannot make a path with broken PMTUD healthy for every response. A response that shrinks below the failure threshold may merely conceal the defect. Restore PMTU feedback or use a robust packetization strategy; then decide separately whether compression is economical.

## 7. A public case: Cloudflare's PMTUD failure

In [Marek Majkowski's Cloudflare engineering report, published February 4, 2015](https://blog.cloudflare.com/path-mtu-discovery-in-practice/), the event is described as occurring the preceding week. A small number of users reaching Cloudflare through IP tunnels, primarily IPv6 over IPv4, could not access its services after an internal network change. Those are the report's scope and date; this was not a claim that all Cloudflare traffic failed or that a specific byte count failed for every user.

The trigger was expanded use of internal BGP equal-cost multipath routing, or ECMP. Cloudflare's TCP packets for one connection were consistently hashed to a server using flow information, but ICMP messages carrying path MTU feedback were hashed using different information. An ICMP message could land on a different server than the one that owned the TCP flow. The feedback therefore failed to update the sender that needed to reduce its packet size. The contributing path condition was tunneling, which reduced the usable MTU for the affected users. The blast radius was limited to those paths, making ordinary tests and many users look healthy.

Cloudflare reported a temporary IPv6 MTU reduction to 1280 bytes and RFC 4821-style probing for IPv4. Its longer-term fix distributed the ICMP MTU messages to the servers so the flow owner could receive the feedback. The 1280 number belongs to Cloudflare's dated mitigation and IPv6 context, not to a general recommendation to set every interface to 1280. The report also published the `pmtud` implementation. A later [Cloudflare follow-up, September 10, 2018](https://blog.cloudflare.com/increasing-ipv6-mtu/), explained that the conservative IPv6 setting was being revisited after the routing solution matured.

This case is useful because it sharpens the phrase "firewall drops ICMP." [RFC 2923](https://www.rfc-editor.org/rfc/rfc2923.html#section-2.1) documented actual ICMP suppression as one route to a black hole. Cloudflare's report demonstrated another: ICMP existed but did not reach the correct flow owner because of ECMP routing. From the TCP sender's point of view, both remove the feedback it needs. The transferable test is to trace the control message all the way back to the correct sender, not merely verify that some router generated one.

Notice how the observation differs from a compression incident. A codec regression would usually appear as more CPU time, a changed `Content-Encoding`, a larger coded body, or a larger delay before the first bytes leave the server. Cloudflare's affected paths were selective by network topology: tunnels made the usable MTU smaller, and ICMP feedback took an incompatible ECMP route. The same application body could succeed for one user and fail for another without the serializer changing. In an incident, this tells us to segment results by client network or tunnel path. A global average can hide a small affected group, exactly the kind of group the report describes.

The report's proposed correction was also layered. Temporarily limiting IPv6 packets to 1280 avoided the immediate oversized-packet condition for those paths. Probing helped IPv4 recover without depending wholly on ICMP. Forwarding PMTU feedback to the proper server addressed the routing fault. These actions solve related but different problems: immediate service restoration, robustness to missing feedback, and correct delivery of feedback. Copying only the conservative MTU would retain the inefficiency, and copying only a probe setting would leave the path control signal broken. The case is a good reminder to separate a mitigation from a repair.

The incident ledger is explicit: organization, Cloudflare; event, late January 2015 as described in a February 4 report; source owner, Cloudflare engineering; mechanism, ECMP routing of ICMP PMTU feedback to a different server from the TCP flow; verified number, 1280-byte temporary IPv6 MTU; lesson, preserve PMTU feedback to the flow owner and use probing as a robustness layer. No invented request latency or affected-user percentage is required to teach it.

## 8. Diagnose by the clock and the boundary

Start with client-visible phases and a controlled body size sweep. If `time_starttransfer` grows while body bytes stay constant, investigate application production, compression queue, proxy buffering, and TLS flushing. If first byte stays stable but `time_total` grows in proportion to bytes, inspect bandwidth, flow control, congestion, and transport loss. If a tiny response works and a large one hangs, add path MTU to the candidate list immediately, but still verify it with packets and counters. If the body shrinks under compression and completion gets worse, compare CPU and queue wait before changing algorithms again.

![A diagnostic tree chooses measurements from first-byte delay, slow transfer, and size-dependent stalls.](/imgs/blogs/payloads-on-the-wire-compression-record-sizes-and-the-mtu-you-forgot-6.webp)

| Observation | Next discriminating measurement | Candidate boundary | A dangerous shortcut | Source |
| --- | --- | --- | --- | --- |
| First byte late, transfer tail short | Application produce and flush timestamps; `curl` `time_starttransfer` | App, compressor, proxy, TLS record | Force a larger TCP MSS | Derived diagnostic workflow here |
| First byte early, tail proportional to bytes | `curl` `size_download`, `time_total`; `ss -tin` | Link goodput, flow control, packet loss | Raise compression level without CPU measurement | Derived diagnostic workflow here |
| Small response works, large response stalls | SYN MSS, repeated sequence ranges, ICMP/PTB, `ss -tin` | PMTU and feedback path | Treat `ping` success as proof of health | [RFC 2923, September 2000](https://www.rfc-editor.org/rfc/rfc2923.html#section-2.1) |
| Serialized bytes fall, HTTP bytes barely move | Raw and coded `wc -c` for the same fixture | Codec plus content coding | Claim the raw codec ratio as wire savings | [Protocol Buffers wire-format guide](https://protobuf.dev/programming-guides/encoding/) and lab below |

For a read-only production first pass on Linux, collect configuration and counters before touching the route or firewall:

```bash
ip route get 203.0.113.10
ip -br link
sysctl net.ipv4.tcp_mtu_probing
ss -tin dst 203.0.113.10
nstat -az | grep -E 'TcpRetransSegs|IpFragFails|IcmpInDestUnreachs'
```

Replace the documentation address with the real destination. `ss` may show `pmtu`, `advmss`, `mss`, retransmissions, and congestion state depending on the kernel and socket. `nstat` counts are host-wide, so correlate them with a short test and capture instead of attributing all increments to one flow. A packet capture with a narrow host and port filter can test the hypothesis, but captures may contain credentials, tokens, personal data, and payloads. Limit duration and access. Do not run mutating route, firewall, or `tc` commands on an unspecified production interface.

For HTTPS, an HTTP symptom does not grant visibility into application bytes at an intermediate capture point. You can still inspect TCP sequence ranges, retransmission timing, packet sizes, and ICMP. An origin or client instrumented at the TLS boundary can add record timing. When a gateway terminates TLS and opens a second connection, inspect both legs. The place where the symptom first appears is more informative than a single end-to-end stopwatch.

### Choose a change only after naming the winning measurement

Compression, serialization, flush policy, and PMTU fixes each have a different success metric. If the problem is expensive egress, the objective is fewer transferred bytes for the same content and acceptable CPU. If the problem is the first renderable byte, the objective is earlier delivery of a useful chunk, even if the final byte count barely changes. If the problem is a large-response stall, the objective is successful packet delivery on the affected path with reliable feedback when routing changes. Those goals should be written next to the proposed change so a later team can tell whether it worked.

For a compression change, collect a baseline on the actual distribution of payloads. Classify content types, size bands, and client `Accept-Encoding` support. Benchmark a candidate at the same offered load and compare both bytes and p99. Watch CPU and memory. Roll out to a small share of traffic if the system permits it, and verify that caches do not serve the wrong representation. A lower byte count with worse p99 is not a success if p99 was the reason for the project.

For a serializer change, start with contracts. Preserve every field and semantic value, verify old and new clients can read the intended versions, then measure raw and content-coded bytes separately. Count headers and envelopes when the message is small. Treat a protobuf win on one synthetic fixture as a prompt for a corpus benchmark, not as a migration decision. The API design choice includes tooling, debuggability, schema evolution, and client support beyond this wire calculation.

For a flush change, instrument the first useful chunk. A flush that sends an empty or unrenderable prefix may improve a metric without improving a user's experience. Measure when the client can do useful work, not only when the first encrypted record exists. Watch whether smaller records increase CPU or reduce bulk throughput. A good first-byte policy may deliberately use small early records and larger later records, provided the application and proxies preserve that behavior.

For a PMTU change, first make the failure path observable. Capture the packet that cannot cross, the feedback that should return, and the sender's response. Check actual encapsulation and direction. A scoped MSS clamp can restore a TCP service when the path cannot be corrected immediately, but keep monitoring because route changes can alter the bottleneck. Protect legitimate ICMP PMTU messages rather than blocking an entire control protocol for neat-looking firewall rules. A permanently low MTU across all users is a cost and a diagnostic blindfold.

These choices can interact. Compression can change the response length, which changes the number of packets but usually not the configured MSS. A flush policy can change record timing, which changes when TCP receives bytes to segment. A protobuf change can shrink the raw body while leaving coded bytes almost the same. The workflow is to change one variable, predict which boundary should move, and instrument that boundary. When the predicted boundary does not move, the theory was wrong or another layer absorbed the change. Either result is useful evidence.

Write the capture point and accounting unit beside every chart. "Response size" without either one can mean decoded JSON in one dashboard and compressed HTTP bytes in another. "Latency" can mean first byte, complete body, or the application's own handler timer. Consistent labels are a small operational habit that prevents hours of arguing over numbers that were never measuring the same boundary.

## Run it yourself

### Question

For one deterministic response fixture, how many body bytes does each coding save, and what transfer-time budget does that create at a chosen goodput? The experiment tests the CPU-for-bytes model, not a universal ranking of codecs. It also measures the raw-versus-coded boundary that a JSON/protobuf comparison must report.

### Preconditions

Use Linux with Python 3, `gzip`, `brotli`, `zstd`, and `wc`. This experiment only creates files under a temporary directory and requires no root access. Check versions because compressor defaults and implementations change. The canonical `netlab` namespaces `c` and `s` from [the introductory setup](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) are useful for a later packet capture but are not required for this byte-count experiment. If you run packet captures or change namespace routes to explore PMTUD separately, use the lab interfaces `c0` and `s0` only and follow the setup's scoped cleanup. Do not put a synthetic MTU fault on a production interface.

### Baseline

```bash
set -euo pipefail
command -v python3 gzip brotli zstd wc
python3 --version
gzip --version | head -1
brotli --version
zstd --version
lab_dir=$(mktemp -d /tmp/payload-wire.XXXXXX)
export lab_dir
python3 - <<'PY'
import json, os
from pathlib import Path
p = Path(os.environ['lab_dir']) / 'events.json'
events = [
    {'id': i, 'service': 'checkout', 'state': 'accepted',
     'region': 'ap-southeast-1', 'message': 'payment accepted'}
    for i in range(1000)
]
p.write_text(json.dumps({'events': events}, separators=(',', ':'),
                        ensure_ascii=False), encoding='utf-8')
PY
wc -c "$lab_dir/events.json"
```

Read the first column of `wc -c`, which is serialized JSON body bytes. The file has a fixed record count and fixed strings; the exact byte count is determined by the code, not by a claimed production sample. Run `wc -c` again if your Python or fixture differs. The expected state is a nonzero file under `lab_dir`, and repeated runs with the same code should have the same byte count. This baseline is the uncoded representation.

### Apply one change

Apply content coding to the *same bytes*. Each command writes to a new file; the source remains fixed. The explicit levels make the comparison reproducible rather than relying on each tool's changing default.

```bash
gzip -6 -c "$lab_dir/events.json" > "$lab_dir/events.json.gz"
brotli -q 5 -c "$lab_dir/events.json" > "$lab_dir/events.json.br"
zstd -q -5 -c "$lab_dir/events.json" > "$lab_dir/events.json.zst"
wc -c "$lab_dir/events.json" "$lab_dir/events.json.gz" \
  "$lab_dir/events.json.br" "$lab_dir/events.json.zst"
```

### Compare

Read `wc -c` for each coded body. On this deliberately repetitive fixture, each coded file should be smaller than the uncoded file; the exact ratio is deliberately left for your tool versions and CPU. Then compute the explanatory transfer-time budget at 10 Mbit/s from the measured byte difference. `10_000_000` is an assumed goodput in this model, not a result from your link.

```bash
python3 - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['lab_dir'])
plain = (p / 'events.json').stat().st_size
for suffix in ('.gz', '.br', '.zst'):
    coded = (p / ('events.json' + suffix)).stat().st_size
    saved_ms = 8 * (plain - coded) / 10_000_000 * 1000
    print(f'{suffix}: plain={plain} coded={coded} '
          f'saved_transfer_ms_at_10Mbit={saved_ms:.3f}')
PY
```

Expected: `coded` is positive and less than `plain` for this fixture, so each computed transfer saving is positive. The exact millisecond result is determined by your actual `wc -c` values. This calculation excludes compression CPU, decompression, TLS, IP overhead, congestion, and queueing. To choose a production coding, add CPU timing under representative concurrency and measure client `time_starttransfer` and `time_total` at matched request load. Changing a compressor and a serializer simultaneously would not isolate either one.

For a JSON/protobuf extension, define the identical logical `Event` and `Batch` data in a `.proto`, use `protoc --encode` to serialize it, then repeat `wc -c` and the same three content codings. The following commands use the same `lab_dir` and run **after** the compression comparison, before Reset. They change only the serialization format for the same fields and event count.

```bash
command -v protoc
protoc --version
cat > "$lab_dir/events.proto" <<'PROTO'
syntax = "proto3";
message Event {
  uint32 id = 1;
  string service = 2;
  string state = 3;
  string region = 4;
  string message = 5;
}
message Batch {
  repeated Event events = 1;
}
PROTO
python3 - <<'PY'
import os
from pathlib import Path
p = Path(os.environ['lab_dir']) / 'events.textproto'
with p.open('w', encoding='utf-8') as f:
    for i in range(1000):
        f.write(f'events {{ id: {i} service: "checkout" '
                'state: "accepted" region: "ap-southeast-1" '
                'message: "payment accepted" }\n')
PY
protoc --proto_path="$lab_dir" --encode=Batch "$lab_dir/events.proto" \
  < "$lab_dir/events.textproto" > "$lab_dir/events.pb"
gzip -6 -c "$lab_dir/events.pb" > "$lab_dir/events.pb.gz"
brotli -q 5 -c "$lab_dir/events.pb" > "$lab_dir/events.pb.br"
zstd -q -5 -c "$lab_dir/events.pb" > "$lab_dir/events.pb.zst"
wc -c "$lab_dir"/events.pb*
protoc --proto_path="$lab_dir" --decode=Batch "$lab_dir/events.proto" \
  < "$lab_dir/events.pb" | grep -c '^events {'
```

Read the `wc -c` columns and the final decoded-event count. Expect `1000` decoded events. On the specified local environment, the protobuf byte counts were 58,870 raw, 2,387 gzip, 1,667 brotli, and 1,502 zstd. Another compiler or compressor version can differ, so record the output rather than treating those numbers as invariants. Compare each coded protobuf file to the coded JSON file made with the *same* compressor setting. The example demonstrates exactly why raw protobuf size is not a sufficient estimate of HTTP body savings.

### Reset

```bash
test -n "${lab_dir:-}"
case "$lab_dir" in /tmp/payload-wire.*) rm -rf -- "$lab_dir" ;; \
  *) printf 'Refusing unexpected lab path: %s\n' "$lab_dir" >&2; exit 1 ;; esac
unset lab_dir
```

The reset removes only the temporary directory that this experiment created. No namespace, route, qdisc, firewall, or sysctl changed. For a production read-only translation, save one representative response with `curl -D headers.txt --raw -o body.bin`, inspect `Content-Encoding` and `Vary`, and count `body.bin` with `wc -c`. Restrict that capture to data you are permitted to store; response bodies can contain secrets and personal information. Compare client timing with `curl -w` only after controlling cache and connection state.

## Key takeaways

- Name the byte count: logical object, serialized body, content-encoded body, TLS record, or IP packet.
- Compression pays when saved transfer time exceeds critical-path compression and decompression time under the actual path goodput and server load.
- `time_starttransfer` and `time_total` split first-byte delay from transfer tail; neither identifies the culprit without a boundary measurement.
- A TLS record can span many TCP segments. Neither an application write nor a record is an IP packet.
- Compare JSON and protobuf with the same logical data before and after the same content coding. The raw message ratio is not automatically the wire ratio.
- A local MTU or jumbo setting is not a path guarantee. A size-dependent stall with repeated large retransmissions warrants a PMTU feedback check.
- Preserve ICMP Packet Too Big delivery to the correct flow owner and use packetization-layer probing as a robustness tool. A small-response success is weak evidence of path health.

## Further reading

- [RFC 9110, HTTP Semantics, June 2022](https://www.rfc-editor.org/rfc/rfc9110.html): content coding and negotiation.
- [RFC 8446, TLS 1.3, August 2018](https://www.rfc-editor.org/rfc/rfc8446.html): record structure and limits.
- [RFC 2923, TCP Problems with Path MTU Discovery, September 2000](https://www.rfc-editor.org/rfc/rfc2923.html): the black-hole problem.
- [RFC 4821, Packetization Layer Path MTU Discovery, March 2007](https://www.rfc-editor.org/rfc/rfc4821.html): probing without depending solely on ICMP.
- [Cloudflare, Path MTU discovery in practice, February 4, 2015](https://blog.cloudflare.com/path-mtu-discovery-in-practice/): a real feedback-delivery failure.
