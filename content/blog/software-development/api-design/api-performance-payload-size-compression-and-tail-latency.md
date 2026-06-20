---
title: "API Performance: Payload Size, Compression, and Tail Latency"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "Make the contract fast by design: where the milliseconds actually go in an API call, how to trim and compress a fat payload, why connection reuse and batch endpoints beat micro-tuning, and why the p99 of a fan-out is the number your customers feel."
tags:
  [
    "api-design",
    "api",
    "rest",
    "http",
    "performance",
    "latency",
    "compression",
    "tail-latency",
    "http2",
    "payload",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-1.png"
---

The incident started with a chart, not an error. The Payments dashboard for our fictional commerce platform showed the `GET /v1/orders` endpoint at a comfortable 28 ms median. Nobody was paging. But the mobile team had filed a ticket that read, in full: "Orders screen takes forever on the train." Forever, it turned out, meant four to nine seconds. The server logs disagreed: the handler returned in 28 ms, every time. The 28 ms was real. It was also almost completely irrelevant to the person staring at a spinner on a moving train.

Two things were happening, and neither showed up in the median server-time graph. First, the endpoint returned the *entire* order object — 142 fields per order, 50 orders per page, line items, full shipping addresses, a denormalized customer blob, a `raw_gateway_response` field nobody outside Payments had ever read — about 200 KB of uncompressed JSON. Over a congested cellular link with maybe 400 Kbps of usable downstream and 150 ms of round-trip time, that 200 KB is not 28 ms of anything. It is several hundred milliseconds of pure transfer, on top of a DNS lookup, a TCP handshake, and a TLS handshake the app paid *again* because the connection had idled out between screens. Second, the screen actually made eleven calls: one to list orders, then ten more to fetch each order's current shipment status from a separate service. Eleven sequential round-trips at 150 ms each is 1.65 seconds of network floor before a single byte of useful work — and you cannot compress a round-trip away.

This post is about making the contract *fast* — fast as a design property, not as an afterthought you bolt on with a profiler the week before launch. We will start by being honest about where the milliseconds go: DNS, TCP, TLS, server time, transfer, parse. Then we will work the levers in order of payoff. **Payload size** first, because over-fetching is the most common and most fixable waste, and the math is simple: transfer time is roughly $\text{bytes} / \text{bandwidth} + \text{RTT}$. **Compression** next — gzip and brotli routinely shrink JSON 5 to 10 times for the cost of some CPU. **Connection reuse**, because the handshake you pay once and amortize across a thousand requests is the handshake you do not pay at all. **Chattiness** — the API-level N+1 that turns one screen into eleven round-trips. And finally **the tail**: why the p99, not the median, is the number that defines your reputation, and why a request that fans out to ten dependencies waits on the *slowest* one, so the tail compounds in a way the average never reveals.

![A vertical stack showing where the milliseconds of an API call go, from DNS lookup and TCP and TLS handshakes through server time, transfer, and client parse](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-1.png)

By the end you will have a mental model — the latency-lever ladder — for deciding *which* optimization to reach for, grounded in the one rule that survives every performance fad: **measure first, then fix the bottleneck.** Compressing a body when the bottleneck is eleven round-trips is wasted effort. Adding a faster serializer when the bottleneck is one slow downstream dependency is wasted effort. The whole point of being a principled performance engineer is knowing, before you touch anything, which lever actually moves the number your caller feels. This is the same spine that runs through the whole series: an API is a contract you are designing for a caller you will never meet — and that caller is often on a train, on a phone, on a flaky link, half a world away from your data center. The contract has to be fast *for them*, not for your localhost benchmark.

## 1. The latency anatomy of an API call

Before you can speed anything up, you have to know where the time goes. "The API is slow" is not a diagnosis; it is a symptom with at least six possible causes, and they live in completely different places. Here is the full round-trip, from the moment a client decides to call you to the moment it has a parsed object in hand.

**DNS resolution.** The client turns `api.example.com` into an IP address. If it is already cached, this is free. If it is cold — a fresh app launch, a new network — it is one or more round-trips to a resolver, anywhere from 5 ms to over 100 ms on a poor mobile link. You do not control the client's resolver, but you do control your DNS TTLs and whether you scatter your API across a dozen hostnames the client has to resolve separately.

**TCP handshake.** A new TCP connection costs one round-trip — the SYN, SYN-ACK, ACK dance — before any application data flows. On a link with 150 ms RTT, that is 150 ms of pure setup. (We will define RTT precisely in a moment: it is the *round-trip time*, the time for a packet to go to the server and a reply to come back.)

**TLS handshake.** On top of TCP, establishing the encrypted channel costs one round-trip in TLS 1.3 (the modern default) or two in TLS 1.2. So a cold HTTPS connection on a 150 ms link is roughly: 150 ms (TCP) plus 150 to 300 ms (TLS) before your server even sees the request line. That is 300 to 450 ms spent on *nothing the user asked for*, paid every time the client opens a fresh connection.

**Server time (time to first byte minus network).** This is the part you usually obsess over and the part that shows up in your dashboards: the time your handler spends parsing the request, hitting the database, serializing the response. For a well-built endpoint this is often the *smallest* slice on a remote link — 5 to 80 ms — which is exactly why server-time graphs lie to you about the user's experience.

**Transfer.** The response bytes travel back over the wire. This is governed by size and bandwidth and is the slice that scales with your payload. We will derive its math next, but the headline is: a fat body over a thin pipe can take longer than everything else combined.

**Client parse and deserialize.** The client turns the bytes into objects — `JSON.parse`, a protobuf decode, building model instances. For a small body this is negligible; for a multi-megabyte JSON array on a low-end phone it can be tens of milliseconds of main-thread blocking that freezes the UI.

The figure above stacks these so you can see the shape of the problem: on a fast data-center link between two services, server time dominates and the rest is noise. On a cold mobile link, the handshakes and transfer dominate and server time is a rounding error. **The same endpoint has two completely different bottlenecks depending on who is calling and over what.** That is the first principle of API performance: *there is no single "the latency" — there is a distribution across callers, networks, and request shapes, and you must optimize the bottleneck for the caller who matters.*

### The principle: a round-trip floor you cannot compress away

Here is the rule that governs everything below, stated rigorously so we can lean on it. The wall-clock time of a single request is approximately:

$$T_{\text{request}} \approx T_{\text{setup}} + n_{\text{rtt}} \cdot \text{RTT} + \frac{\text{bytes}}{\text{bandwidth}} + T_{\text{server}} + T_{\text{parse}}$$

where $T_{\text{setup}}$ is DNS plus handshakes (paid once per connection, then amortized), $n_{\text{rtt}}$ is the number of *application-level* round-trips the interaction requires, RTT is the round-trip time of the link, and the $\text{bytes}/\text{bandwidth}$ term is transfer.

The term that traps people is $n_{\text{rtt}} \cdot \text{RTT}$. Notice it has nothing to do with payload size or compression. Each round-trip is a fixed tax set by the speed of light and the physical distance between client and server. London to a US-East data center is roughly 75 to 90 ms one way, so an RTT near 150 to 180 ms, *no matter how fast your code is*. This is the API-design equivalent of Amdahl's law: you can drive the server-time and transfer terms toward zero, but if your design forces eleven sequential round-trips, you are pinned at eleven times RTT. The only way past it is to reduce $n_{\text{rtt}}$ — batch, reuse connections, multiplex — not to optimize the work *inside* each round-trip. We will come back to this floor repeatedly; it is the reason chattiness is the most expensive design mistake on this list.

### Measuring the breakdown, not just the total

The reason the opening incident dragged on for two days is that the team had exactly one number — server time, 28 ms — and that number was *true* and *useless*. The discipline that would have caught it in five minutes is to instrument each slice of the equation separately and look at the *shape* of the breakdown, not the sum. A browser exposes this directly through the Resource Timing and Navigation Timing APIs, which break a request into `domainLookupStart`/`domainLookupEnd` (DNS), `connectStart`/`connectEnd` (TCP), `secureConnectionStart` (TLS), `requestStart`/`responseStart` (server time, the time to first byte), and `responseStart`/`responseEnd` (transfer). On the command line, `curl` will print the same breakdown with a custom write-out format, which is the fastest way to see where a slow call actually spends its time:

```bash
curl -s -o /dev/null -w \
  'dns=%{time_namelookup}s tcp=%{time_connect}s tls=%{time_appconnect}s ttfb=%{time_starttransfer}s total=%{time_total}s\n' \
  'https://api.example.com/v1/orders'
```

A run of that against the fat endpoint over a throttled link would have printed something like `dns=0.04s tcp=0.19s tls=0.37s ttfb=0.40s total=4.55s` — and the story leaps off the screen: time-to-first-byte (which includes the server's 28 ms plus the full handshake) was 0.40 s, but `total` was 4.55 s, so **more than four seconds was pure transfer after the first byte arrived.** No amount of staring at the server-time dashboard would have shown that, because the server-time dashboard stops counting at the moment the first byte leaves the building. The general principle is worth stating plainly: *the server measures what it can see, and it cannot see the wire.* If your only latency signal is generated inside your handler, you are blind to the largest slice of a remote caller's experience. The fix is real-user monitoring (RUM) — collecting the client-side breakdown from actual devices on actual networks — so that the p99 you alert on is the p99 your users live, not the p99 of your handler. We will return to this when we talk about the tail, because measuring the tail correctly is the entire prerequisite to taming it.

## 2. Payload size: the bytes you send are the time you pay

The cheapest performance win in most APIs is to stop sending bytes nobody reads. Over-fetching — returning the full, denormalized representation of a resource when the caller wanted three fields — is the single most common waste in REST APIs, and it costs on every axis: transfer time on the wire, parse time on the client, memory on both ends, and serialization CPU on the server.

Let us make the transfer cost precise. The transfer time for a response is approximately:

$$T_{\text{transfer}} \approx \frac{\text{size}}{\text{bandwidth}} + \text{RTT}$$

The RTT term is there because even a tiny body needs at least one round-trip, and because TCP's slow-start means the first few round-trips after a connection opens cannot use the full bandwidth — the congestion window has to grow. For a body that fits in the initial congestion window (roughly 14 KB on modern stacks), you pay about one RTT and you are done. For a body much larger than that, you pay $\text{size}/\text{bandwidth}$ plus several RTTs while the window ramps. **The practical consequence: the difference between a 10 KB body and a 200 KB body on a slow link is not 20 times the transfer time — it is often more, because the small body fits in the first window and the large one does not.**

#### Worked example: trimming a 200 KB order list on 4G

Take the real shape of the incident from the intro. The `GET /v1/orders` endpoint returns 50 orders, each fully expanded, for about 200 KB of uncompressed JSON. The mobile Orders screen actually displays four things per order: an ID, a status, a total, and a date. Everything else — line items, the customer blob, the gateway response — is dead weight on this screen.

Model a congested 4G link as roughly 400 Kbps (50 KB/s) usable downstream and 150 ms RTT. Frame these as approximate; real cellular varies wildly. The fat payload:

$$T_{\text{transfer}} \approx \frac{200\text{ KB}}{50\text{ KB/s}} + 0.15\text{ s} = 4.0\text{ s} + 0.15\text{ s} \approx 4.15\text{ s of transfer}$$

That is the "forever" the mobile team reported, and the 28 ms server time is invisible inside it. Now project to just the four fields the screen uses — call it 60 KB for 50 orders, because IDs and timestamps and totals are small:

$$T_{\text{transfer}} \approx \frac{60\text{ KB}}{50\text{ KB/s}} + 0.15\text{ s} = 1.2\text{ s} + 0.15\text{ s} \approx 1.35\text{ s}$$

Projection alone cut transfer by about 3x. We have not even compressed yet — that comes in the next section and stacks on top. The lever here is **sparse fieldsets and sensible page sizes**, which this series covers in depth in [filtering, sorting, and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql) and [pagination strategies](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale). The performance argument for those features is exactly this: they let the caller pay for the bytes they use.

Here is what the wire looks like when the client asks for only what it needs, using a sparse-fieldset query parameter:

```http
GET /v1/orders?fields=id,status,total,created_at&page_size=50 HTTP/2
Host: api.example.com
Authorization: Bearer <token>
Accept: application/json
```

```http
HTTP/2 200 OK
Content-Type: application/json
Content-Length: 61240
Cache-Control: private, max-age=15

{
  "data": [
    { "id": "ord_1a2b", "status": "shipped", "total": "49.99", "created_at": "2026-06-19T10:02:00Z" },
    { "id": "ord_3c4d", "status": "pending", "total": "129.00", "created_at": "2026-06-19T11:14:00Z" }
  ],
  "page": { "next_cursor": "eyJpZCI6Im9yZF8zYzRkIn0" }
}
```

Two design notes that matter for performance, not just cleanliness. First, the `total` is a string, not a float, because money in floats is a bug — but a string also compresses better than you might fear, since gzip loves repeated digit patterns. Second, the `next_cursor` is opaque and short; it is not a re-serialization of the whole query state. Bloated cursors are a sneaky payload tax that grows with every page.

### Over-fetching costs on four axes, not one

It is tempting to think of payload size as purely a transfer problem, but a fat body taxes you in four distinct places, and naming them helps you justify the work to a skeptical reviewer who thinks "the network is fast, who cares about 200 KB."

First, **transfer time** on the wire, which we just quantified — the most visible cost, and the only one most people think about. Second, **server serialization CPU**: turning 200 KB of objects into JSON text costs real CPU per request, and at thousands of requests per second that is cores you are paying for to produce bytes nobody reads. Third, **client parse and memory**: a low-end phone parsing a multi-hundred-kilobyte JSON array blocks its main thread for tens of milliseconds and allocates a large object graph, which on a memory-constrained device can trigger garbage collection pauses or even an out-of-memory kill of the app. Fourth, **caching efficiency**: a fat, denormalized body that mixes slowly changing data (the order total) with fast-changing data (the live shipment status) cannot be cached well, because the whole thing invalidates the moment the fastest-changing field changes. A leaner, well-factored body lets you cache the stable parts aggressively.

So "stop over-fetching" is not a micro-optimization; it is a four-way win that compounds. The mechanism — letting the caller declare which fields and which related resources it wants — is the subject of [filtering, sorting, and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql), and the performance case for it is exactly these four axes.

### Pagination is a payload lever, not just a UX one

The reason you bound page sizes is not only to keep clients from accidentally requesting 50 million rows; it is that an unbounded page is an unbounded payload, and an unbounded payload is an unbounded transfer time and an unbounded server serialization cost. A `page_size` cap of, say, 100 with a default of 25 is a *performance contract*: it promises the caller that no single response will be catastrophically large, and it protects your server from spending 4 seconds serializing a response the client will time out on anyway. For very large result sets, the answer is not a bigger page — it is **streaming**, which we cover at the end of this post: send the rows as you produce them so the client can start parsing before the server has finished generating, and neither side has to hold the whole thing in memory.

## 3. Compression: trade CPU for bytes

Once you have stopped sending bytes nobody reads, compress the bytes that remain. This is the highest-leverage one-line change in API performance: turn on response compression at the server or gateway and JSON bodies typically shrink 5 to 10 times, because JSON is extraordinarily compressible — it is full of repeated keys (`"status"`, `"created_at"`), repeated structural characters, and predictable value patterns.

Compression on the web works through **content negotiation**, the same Accept/Content-Type mechanism this series covers in [content negotiation](/blog/software-development/api-design/content-negotiation-media-types-and-representations). The client advertises what it can decode; the server picks one and tells the client what it used:

```http
GET /v1/orders?fields=id,status,total,created_at HTTP/2
Host: api.example.com
Accept: application/json
Accept-Encoding: br, gzip
```

```http
HTTP/2 200 OK
Content-Type: application/json
Content-Encoding: gzip
Vary: Accept-Encoding
Content-Length: 9180

...gzip-compressed bytes...
```

Three headers carry the whole protocol. `Accept-Encoding: br, gzip` is the client saying "I can handle brotli or gzip." `Content-Encoding: gzip` is the server saying "here is gzip; decode it." And `Vary: Accept-Encoding` is the load-bearing one for caching — it tells every cache between you and the client that the response *depends on* the requested encoding, so a cache must not hand a gzip body to a client that only asked for identity. Forget `Vary` and you will eventually serve a brotli body to a client that cannot decode it, which looks exactly like data corruption and is miserable to debug.

### gzip versus brotli: the CPU-versus-bytes trade

Gzip (the DEFLATE algorithm, around since the 1990s) is the safe, universal default: every HTTP client understands it, it is fast to encode, and it shrinks JSON about 5 to 8 times at a middle compression level. Brotli, developed at Google and standardized for HTTP, generally compresses 10 to 20 percent smaller than gzip on text at comparable speed, and dramatically smaller at its highest quality levels — but those high levels are *slow* to encode. The trade is the whole story, and it is captured in the comparison below.

![A matrix comparing no compression, gzip, and two brotli levels across JSON compression ratio, encoder CPU cost, and the best use case for each](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-2.png)

| Option | Typical JSON ratio | Encode CPU | Decode | Best for |
| --- | --- | --- | --- | --- |
| none (`identity`) | 1x | none | none | already-tiny bodies, already-compressed data |
| gzip level 6 | 5x to 8x | low | fast | dynamic JSON responses (the default) |
| brotli level 5 | 6x to 9x | low to mid | fast | a balanced always-on default |
| brotli level 11 | 7x to 11x | high | fast | cacheable static assets, encode once |

The decision rule falls out of the table. For **dynamically generated responses** — your `/orders` list, which is different for every caller — use gzip at a middle level or brotli at a low-to-mid level. You are encoding on the hot path, once per request, and you cannot afford brotli-11's CPU there. For **cacheable assets** — an OpenAPI spec, a JavaScript SDK bundle, a static price list — use brotli-11: you encode once, cache the compressed bytes, and serve them a million times, so the CPU is amortized to nothing and you keep the smallest possible body forever.

#### Worked example: gzip on the trimmed order list

Continue the 4G example. We trimmed the body to 60 KB. JSON like this — repetitive keys, short values — compresses well; assume a conservative 6.5x gzip ratio, so about 9 KB on the wire. Transfer time:

$$T_{\text{transfer}} \approx \frac{9\text{ KB}}{50\text{ KB/s}} + 0.15\text{ s} = 0.18\text{ s} + 0.15\text{ s} \approx 0.33\text{ s}$$

Stack the two levers and the picture is dramatic. We went from 4.15 s (fat, uncompressed) to 1.35 s (trimmed) to about 0.33 s (trimmed plus gzipped) — and most of that 0.33 s is now the unavoidable RTT, not transfer. The before-and-after is worth seeing as a single image, because it is the canonical shape of a mobile payload win.

![A two-column before and after figure contrasting a 200 KB uncompressed payload at about 400 ms transfer against a 60 KB sparse body gzipped to about 9 KB at under 30 ms](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-3.png)

The CPU cost is real but small for gzip on a body this size — sub-millisecond to a couple of milliseconds of server time to compress 60 KB at level 6, against hundreds of milliseconds saved on the wire. That is a trade you take every time on a remote link. Where the trade flips is the next point.

The break-even is worth reasoning about explicitly, because it explains why "compress everything at maximum quality" is wrong. Compression is a trade of server CPU time for client transfer time, and it pays only when the transfer time saved exceeds the CPU time spent — weighted by the fact that CPU is your scarce, shared resource and the client's transfer time is theirs. On a fast internal link between two services in the same data center (bandwidth in gigabits, RTT under a millisecond), compressing a 60 KB body might save microseconds of transfer while costing a millisecond of CPU — a *losing* trade, which is why some high-throughput internal APIs skip compression entirely and lean on the fat pipe. On the 4G link, the same compression saves hundreds of milliseconds for the same millisecond of CPU — an overwhelming win. **The same body, the same algorithm, opposite conclusions, because the bottleneck moved.** This is the measure-first principle made concrete: there is no universal "always compress" rule, only "compress when transfer is your bottleneck," which for any internet-facing API it almost always is, and for fast internal service-to-service traffic it often is not.

### Do not double-compress, and do not compress the incompressible

Two honest caveats keep you from sabotaging yourself.

First, **do not double-compress.** If your gateway compresses responses, do not also compress in the application, and never compress something that is already compressed — images (PNG, JPEG, WebP), video, a gzipped tarball, a `.zip`. Running gzip over already-compressed bytes burns CPU and frequently makes the output *larger*, because the entropy is already high and you are just adding DEFLATE's framing overhead. A well-configured gateway keys compression on `Content-Type` and skips anything in the already-compressed set.

Second, **tiny bodies are not worth compressing.** A 200-byte `problem+json` error or a 204 with no body gains nothing from gzip; the compressed form can be larger than the original because of the gzip header and trailer (roughly 18 bytes of overhead). Most servers and gateways have a minimum-size threshold — commonly around 1 KB — below which they skip compression. Leave it on.

Third, and this is a security note rather than a performance one: compressing responses that mix attacker-controlled and secret data over a channel an attacker can observe enabled the historical BREACH and CRIME attacks. The practical mitigation for an API is straightforward — do not reflect attacker-controlled input into a response alongside secrets, keep CSRF-style tokens out of compressed bodies, and you are clear. It is worth knowing the failure mode exists; it almost never blocks turning on compression for ordinary JSON.

## 4. Connection reuse: pay the handshake once

Recall the latency anatomy: a cold connection costs DNS plus a TCP handshake plus a TLS handshake — 300 to 450 ms of pure setup on a 150 ms link, before your server sees anything. If a client opens a fresh connection for every request, it pays that tax *every single time*. The fix is **connection reuse**: keep the connection open and run many requests over it, so the handshake is amortized across hundreds or thousands of requests instead of paid per request.

### Keep-alive: amortizing the handshake

HTTP/1.1 made persistent connections the default — `Connection: keep-alive` is implied — so a client *can* reuse a connection for sequential requests. The catch is that HTTP/1.1 allows only one in-flight request per connection at a time. If the client wants concurrency, it opens multiple connections (browsers historically cap around six per host), and each of those still pays its own handshake. So keep-alive amortizes the handshake for *sequential* traffic but does nothing for the head-of-line problem when you want many requests at once.

The amortization math is simple and motivating. Suppose the handshake costs $H \approx 350$ ms and you make $k$ requests. With a fresh connection per request the setup cost is $k \cdot H$. With one reused connection it is $H$ total — the per-request share is $H/k$, which goes to zero as $k$ grows. For a mobile session making 40 requests, that is the difference between 14 seconds of cumulative handshake and 0.35 seconds. **Connection reuse is not a micro-optimization; it can be the largest single factor in a chatty client's total time.**

### HTTP/2 multiplexing: killing app-layer head-of-line blocking

HTTP/2 changes the game by **multiplexing** many concurrent requests over a single connection. Each request/response is a *stream* with its own ID; frames from different streams are interleaved on the one connection, so ten requests can be in flight at once without ten connections and without ten handshakes. This kills *application-layer* head-of-line blocking: in HTTP/1.1, a slow response blocks the connection behind it; in HTTP/2, a slow stream does not block the others because they are independent.

The subtlety — and it is the reason HTTP/3 exists — is that HTTP/2 still runs over a single TCP connection, and TCP is an ordered byte stream. If one TCP packet is lost, TCP holds back *every* later byte until the lost packet is retransmitted, even bytes belonging to other, unrelated streams. So HTTP/2 removes head-of-line blocking at the HTTP layer but reintroduces it at the TCP layer on a lossy link. On a clean data-center network this never bites; on a congested mobile network with packet loss, it can.

### HTTP/3 and QUIC: removing the last floor

HTTP/3 runs over **QUIC**, a transport built on UDP that implements its own per-stream reliability. Because each stream has independent loss recovery, a lost packet on one stream does not stall the others — true multiplexing with no transport-level head-of-line blocking. QUIC also folds the transport and crypto handshakes together, so connection setup is one round-trip, and a resumed connection can be zero round-trips (0-RTT) with the request riding along in the first packet. On lossy, high-latency links — exactly the mobile case that started this post — HTTP/3 is a measurable win; on a clean wired link, HTTP/2 and HTTP/3 are close.

![A matrix comparing HTTP/1.1, HTTP/2, and HTTP/3 across multiplexing, head-of-line blocking, and handshake cost](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-4.png)

| Dimension | HTTP/1.1 | HTTP/2 | HTTP/3 (QUIC) |
| --- | --- | --- | --- |
| Concurrency | one request per connection | many streams, one connection | many streams, one connection |
| App-layer head-of-line | yes, slow response blocks | no, streams independent | no, streams independent |
| Transport head-of-line | per connection | yes, one TCP stream | no, per-stream over QUIC |
| Handshake | TCP + TLS (cold) | TCP + TLS (cold) | QUIC: 1-RTT, or 0-RTT resumed |
| Header overhead | plaintext, repeated | HPACK compressed | QPACK compressed |

For an API designer, the practical guidance is: **prefer HTTP/2 as your baseline** (almost every modern gateway, load balancer, and client supports it), enable HTTP/3 where your edge and clients support it for the mobile win, and on the server-to-server side make sure your HTTP client library reuses connections via a **connection pool**. A connection pool keeps a set of warm, handshake-paid connections to each upstream and hands them out for new requests — the service-to-service analog of browser keep-alive, and a frequent source of accidental slowness when misconfigured (a pool of size one serializes everything; a pool that does not reap idle connections leaks them).

The most common connection-pool failure is also the most insidious because it only appears under load. A client library that defaults to creating a fresh connection per call — which several popular HTTP clients did historically unless you explicitly reused a session object — pays the full handshake on every request. On a localhost benchmark with a sub-millisecond RTT, the handshake is invisible and the code looks fast; in production across a real network, the same code pays 300-plus milliseconds of setup per call and the latency mysteriously triples. The fix is one line — reuse a session or client object across requests rather than constructing one per call — and it is the single most common "why is my service slow" answer in a code review. The general rule: **construct the HTTP client once and share it; never per request.** A pool sized to your concurrency, with sane idle-reaping and a connection lifetime cap (so you do not pin a connection to a backend that has since been replaced behind a load balancer), turns the handshake from a per-request tax into a one-time startup cost. This connection-pooling discipline is closely tied to service-to-service load balancing, which the microservices layer of these systems owns; the API-design takeaway is narrow and firm: a fresh connection per request is a self-inflicted tail.

This connection-reuse story also lives partly at the gateway, which terminates TLS once at the edge and reuses warm backend connections to your services — see [API gateways and the BFF pattern](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern) for how the gateway becomes the place you centralize TLS termination, HTTP/2 and HTTP/3 support, and connection pooling.

## 5. Chattiness: the N+1 at the API level

We have shrunk the payload, compressed it, and reused the connection. None of that touches the most expensive mistake in the intro: the Orders screen made *eleven* round-trips. This is the **N+1 problem at the API level** — one call to fetch a list, then N calls to fetch a detail for each item — and it is the design pattern that pins you against the round-trip floor we derived in section 1.

Recall the floor: $n_{\text{rtt}} \cdot \text{RTT}$. Eleven sequential round-trips at 150 ms is 1.65 seconds you cannot compress, cannot shrink, cannot cache away, because it is the network refusing to break the speed of light eleven times. The bytes might be tiny; the *count* is the killer.

![A two-column before and after figure contrasting eleven sequential round-trips against a single batch request that collapses the round-trip floor](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-5.png)

There are three standard cures, and which you reach for depends on the shape of the problem.

**Batch endpoints.** Add an endpoint that does the join server-side and returns the composite in one round-trip. Instead of "list orders, then for each order fetch its shipment," expose `GET /v1/orders?include=shipment` so the server fetches the shipments in one efficient query (with the database doing the join — see [B-trees and how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) for why a covering index makes that join cheap) and returns them embedded. One round-trip, server does the work.

```http
GET /v1/orders?include=shipment&fields=id,status,total,shipment.eta HTTP/2
Host: api.example.com
Authorization: Bearer <token>
```

```json
{
  "data": [
    {
      "id": "ord_1a2b",
      "status": "shipped",
      "total": "49.99",
      "shipment": { "eta": "2026-06-22", "carrier": "acme" }
    }
  ]
}
```

**GraphQL.** When clients legitimately need *different* compositions for different screens, a query language lets the client ask for the exact graph it needs in one request — the over-fetching and round-trip problems solved together. The cost is that GraphQL invites its own N+1 *inside* the server, at the resolver level, which you defuse with a dataloader that batches per-request lookups into one query. We cover that trap in detail in [GraphQL and the N+1 trap](/blog/software-development/api-design/graphql-the-query-language-schema-and-the-n-plus-one-trap); the performance point here is that GraphQL moves the round-trip count from N+1 to 1 *on the wire*, which is the whole game on a high-latency link.

**A Backend-for-Frontend (BFF).** Put a thin service next to your edge whose job is to compose the calls your client needs and return one tailored payload. The BFF talks to your fleet over a fast internal network where round-trips are cheap (sub-millisecond RTT), and talks to the mobile client over the expensive link with one round-trip. You move the chattiness from the slow link to the fast one. This is the gateway/BFF pattern from [API gateways and the BFF pattern](/blog/software-development/api-design/api-gateways-routing-auth-rate-limiting-and-the-bff-pattern).

#### Worked example: collapsing 11 round-trips into 1

The Orders screen on 4G with 150 ms RTT, before:

$$T_{\text{floor}} = 11 \cdot 0.15\text{ s} = 1.65\text{ s}$$

That is the floor; add transfer and server time on top. After moving to a single batch endpoint with `include=shipment`:

$$T_{\text{floor}} = 1 \cdot 0.15\text{ s} = 0.15\text{ s}$$

We removed 1.5 seconds of pure round-trip tax with a design change that touched zero bytes of compression and zero lines of serializer code. This is why **cutting round-trips sits at the top of the latency-lever ladder**: each round-trip is a floor you cannot reach by any other means. The honest caveat: do not over-rotate into a single mega-endpoint that returns everything for every screen — you will recreate the over-fetching problem you just solved. The right granularity is "one round-trip per user-meaningful action," composed at a layer where round-trips are cheap.

## 6. The tail: why p99 is the number that matters

Here is the part that separates engineers who have run a service at scale from those who have only run it on their laptop. The number your customers experience is not your median latency. It is your **tail** — the p99, the p99.9 — the slowest 1 percent or 0.1 percent of requests. And at scale, the tail does not just exist; it *dominates*, for a reason that is pure probability.

First, definitions, because percentiles confuse people. The **p50** (median) is the latency that half of requests beat. The **p99** is the latency that 99 percent of requests beat — meaning 1 in 100 is *slower* than this. The **p99.9** is the latency 1 in 1,000 requests exceed. These are not exotic edge cases you can ignore. A user who loads ten pages in a session has made dozens of API calls; at p99, the probability that *at least one* of them hits the tail is high. And the slow one is the one they remember.

### The principle: fan-out amplifies the tail

Now the part that compounds. Suppose a request fans out to $n$ backend calls — the gateway calls Orders, Payments, Risk, Inventory, and six more — and the response cannot be assembled until *all* of them return. The latency of the overall request is the **maximum** of the $n$ call latencies, not the average. And the maximum is governed by ruthless probability.

Let $p$ be the probability that a single call comes back *fast* (under some threshold, say the p99 of that service). The probability that *all $n$ calls are fast* is:

$$P(\text{all fast}) = p^{n}$$

assuming independence. Plug in numbers. If each individual call meets its p99 — so $p = 0.99$, only a 1 percent chance of being slow — then for a fan-out of $n = 10$:

$$P(\text{all fast}) = 0.99^{10} \approx 0.904$$

So even though each dependency is fast 99 percent of the time, the *combined* request is fast only about 90 percent of the time — meaning roughly 1 in 10 of these fan-out requests hits *some* dependency's tail. Push to $n = 100$ (a request that touches a hundred shards or services, which large systems do):

$$P(\text{all fast}) = 0.99^{100} \approx 0.366$$

Now nearly two-thirds of requests are slow. **This is the core insight of Dean and Barroso's "The Tail at Scale": a service whose individual p99 is excellent can have a terrible *effective* p99 once it fans out, because the tail of each dependency compounds.** The more you parallelize, the more the slowest straggler defines you. Your median can be beautiful and your users still suffer, because each user request is the *max* of many.

![A graph showing a gateway fanning out to four dependency services with different p99 latencies, then merging at a join node where the slowest dependency sets the response latency](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-6.png)

#### Worked example: one slow dependency sets the p99

Take the payment-authorization request in our platform. The gateway fans out, in parallel, to four services to assemble the response:

- Orders service: p99 of 30 ms
- Payments service: p99 of 35 ms
- Inventory service: p99 of 25 ms
- Risk-scoring service: p99 of 280 ms

The response waits for all four. The overall p99 is dominated by the worst dependency: it cannot be better than about 280 ms at the p99, *no matter how fast the other three are*. You could shave Orders from 30 ms to 5 ms and the user would feel nothing, because Risk is still 280 ms. The figure above makes the merge explicit: four arrows into a join, and the join waits for the slowest.

This is the single most important diagnostic skill in API performance: **find the dependency that owns your tail and fix that, not the fast ones.** The probability math tells you where to look — and your tracing tells you which span is the long pole. This is why observability is not optional for performance work; you cannot fix a tail you cannot see. Tracing each fan-out call with a shared correlation ID, then looking at the *slowest span per request*, is how you find the Risk service in the wild. See [observability for APIs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos) for the metrics, traces, and SLOs that make the tail visible — measure first, then fix the bottleneck.

### Why the average lies, concretely

One more reason to distrust the average. Suppose 99 percent of requests take 20 ms and 1 percent take 1,000 ms (a tail caused by, say, a cold cache or a GC pause). The average is $0.99 \cdot 20 + 0.01 \cdot 1000 = 19.8 + 10 = 29.8$ ms — a number that looks healthy and hides the fact that 1 in 100 of your users waited a full second. Now fan that request out to 5 such services and the chance a user hits at least one 1,000 ms straggler is $1 - 0.99^5 \approx 4.9\%$ — nearly 1 in 20 page loads stalls for a second. The average said 30 ms; reality said 1 in 20 of your customers had a bad time. **Report and alert on percentiles, not averages.**

### Where tails come from, so you can hunt them

Knowing the tail dominates is half the battle; the other half is knowing where the slow stragglers come from, because the source dictates the cure. The usual suspects, roughly in order of how often they bite an API:

- **Queueing under load.** A request that arrives when the server's work queue is momentarily deep waits behind the backlog. Queueing delay is the most common tail source and the most counter-intuitive: a server at 80 percent utilization has a dramatically worse tail than one at 50 percent, because queueing delay rises non-linearly as you approach saturation. This is why headroom is a performance feature, not waste.
- **Garbage collection and runtime pauses.** A stop-the-world GC pause, a JIT recompilation, or a page fault freezes a request for tens to hundreds of milliseconds. These hit *randomly*, so they show up only in the tail, never the median.
- **Cold caches and cache misses.** The 99 requests that hit a warm cache are fast; the one that misses pays the full backend cost. A cache with a 99 percent hit rate has a p99 set entirely by the 1 percent of misses.
- **Resource contention.** A noisy neighbor on a shared host, a lock held a beat too long, a connection-pool exhaustion that makes a request wait for a free connection — all produce occasional slowness invisible to the average.
- **Retries and timeouts of *your* dependencies.** A downstream call that times out and retries adds a full timeout's worth of latency to that one request, landing it squarely in your tail.

The reason to enumerate these is that the cure is specific to the cause: queueing wants headroom and load shedding; GC wants tuning or a different allocation pattern; cold caches want warming or a longer TTL; contention wants isolation and a bigger pool. You find *which* one is your long pole by tracing the slowest requests — not the median ones — and looking at where their time went. A trace of a p50 request tells you nothing about your tail; a trace of a p99.9 request tells you everything. This is the single most actionable habit in tail work: **sample and trace the slow requests specifically**, because they are a different population from the fast ones, with a different bottleneck.

## 7. Tail-tolerance techniques: living with the slow straggler

You cannot make a distributed system's tail zero — there will always be a GC pause, a cold cache, a noisy neighbor, a network hiccup. The discipline is **tail tolerance**: designing so that a single slow component does not become a slow *response*. Dean and Barroso's paper is the canonical reference; here are the techniques that matter for an API designer.

### Hedged and backup requests

The headline technique from "The Tail at Scale." If a request to a replica exceeds a threshold — say the service's p95 — send a *second* copy of the request to a different replica and take whichever responds first, cancelling the loser. The insight is statistical: it is unlikely that *both* the primary and the backup hit their tail on the same request, so the effective latency collapses toward the median while you add only a small amount of extra load (you only hedge the slow tail, not every request, so the overhead is a few percent, not 100 percent).

![A left-to-right timeline of a hedged request that sends a backup to a second replica after passing p95 and uses whichever answer returns first](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-7.png)

The figure traces it: fire the primary at $t=0$, watch the clock, and once you cross the p95 (say 40 ms) without an answer, fire a hedge to replica B. If B answers at 55 ms, you use that and cancel the primary. A rare 280 ms tail becomes a near-p50 response. The two non-negotiable rules: **the hedged request must be safe to send twice** — so hedge only idempotent operations (GET, or a write carrying an idempotency key, which this series covers in [idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions)) — and you must **cancel the loser** so you do not double the real work on every hedge.

```python
async def hedged_get(client, url, hedge_after_ms=40, deadline_ms=300):
    # Fire the primary; if it has not answered by the p95, fire a backup.
    primary = asyncio.create_task(client.get(url))
    done, _ = await asyncio.wait({primary}, timeout=hedge_after_ms / 1000)
    if primary in done:
        return primary.result()

    backup = asyncio.create_task(client.get(url))
    tasks = {primary, backup}
    done, pending = await asyncio.wait(
        tasks, timeout=(deadline_ms - hedge_after_ms) / 1000,
        return_when=asyncio.FIRST_COMPLETED,
    )
    for task in pending:
        task.cancel()  # cancel the loser so we do not double the work
    if not done:
        raise TimeoutError("both primary and hedge exceeded the deadline")
    return next(iter(done)).result()
```

### Timeouts, retries, and budgets

Every outbound call needs a **timeout** — a request with no deadline is a request that can hang forever and pin a thread or a connection until the whole service runs out. But a naive timeout-plus-retry makes the tail *worse*: if a service is slow because it is overloaded, every client timing out and retrying pours *more* load onto the dying service, a feedback loop called a retry storm. Two disciplines fix this:

- **Retry budgets.** Cap retries as a fraction of total traffic (for example, "retries may not exceed 10 percent of requests"). When a service is healthy, retries are rare and the budget is never touched; when it is sick, the budget throttles the retry amplification so you stop kicking it while it is down.
- **Deadline propagation.** Pass a single end-to-end deadline through the fan-out — gRPC builds this in with deadlines, and you can do it with a header in REST. If the overall request has 300 ms left and a call has already burned 250 ms, the downstream call gets a 50 ms deadline, not a fresh 300 ms. Without propagation, a deep call chain multiplies timeouts and a "300 ms" API can legally take 3 seconds.

When you do retry, use **exponential backoff with jitter** so retries spread out instead of synchronizing into a thundering herd. This is the same machinery the message-queue world uses for redelivery; see [dead-letter queues, retries, and exponential backoff](/blog/software-development/message-queue/dead-letter-queues-retries-exponential-backoff) for the broker-side version of the same idea.

### Circuit breakers and load shedding

A **circuit breaker** stops you from hammering a dependency that is clearly down. After a threshold of consecutive failures or timeouts, the breaker "opens" and fails fast — returning an error or a cached fallback *immediately* instead of waiting for another timeout. After a cool-down it goes "half-open," lets a trial request through, and closes again if that succeeds. The breaker converts a slow cascading failure (every request waiting the full timeout on a dead service) into a fast, contained one, which keeps your own latency bounded and gives the sick dependency room to recover.

**Load shedding** is the breaker pointed inward: when *your* service is over capacity, deliberately reject some requests fast — with a `503 Service Unavailable` and a `Retry-After` header — rather than accepting everything and slowing to a crawl for everyone. Shedding the marginal request keeps the requests you *do* accept fast. A queue that grows without bound is just latency you have not measured yet; a bounded queue with shedding is honest about capacity.

```http
HTTP/2 503 Service Unavailable
Content-Type: application/problem+json
Retry-After: 2

{
  "type": "https://example.com/probs/overloaded",
  "title": "Service temporarily overloaded",
  "status": 503,
  "detail": "Shedding load to protect latency. Retry after the indicated delay."
}
```

The `Retry-After` header turns shedding into a contract: the client knows to back off for 2 seconds rather than retrying instantly and making it worse. (This is the same `429`/`503` plus `Retry-After` machinery as rate limiting; the difference is *intent* — 429 means "you are over your quota," 503 with Retry-After means "I am over my capacity.")

### Latency as part of the contract: SLOs and budgets

Everything in this post points back to the series' spine: an API is a contract. Performance is part of that contract whether you write it down or not — your callers *will* form expectations about how fast you are, build timeouts around them, and break when you regress. The mature move is to make the performance contract explicit with a **Service Level Objective (SLO)**: a stated target like "99 percent of `GET /v1/orders` requests complete within 200 ms, measured at the gateway." An SLO is stated in percentiles precisely because, as we derived, the percentile is what callers feel and the average hides the tail. It gives you an *error budget* — the 1 percent you are allowed to miss — which becomes a principled way to decide when to stop optimizing (you are within budget; ship features) and when to drop everything (you are burning the budget; the tail is the priority).

The SLO also disciplines your callers. When a downstream team integrates with you, the SLO tells them what timeout to set: a timeout below your stated p99 will fire on healthy requests and cause spurious retries; a timeout far above it lets a genuinely hung request pin their resources. Publishing "p99 is 200 ms" lets them set a 400 ms timeout with confidence. And the SLO must be measured where the *caller* experiences it — at the gateway or, better, on the client via real-user monitoring — not inside your handler, for exactly the reason the opening incident taught us: the handler cannot see the wire. The machinery for defining, measuring, and alerting on these SLOs is the subject of [observability for APIs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos); the design point here is that **a latency target belongs in your API contract alongside the schema and the status codes**, because a correct response that arrives too late is, to a caller with a timeout, indistinguishable from a failure.

## 8. Server-side performance: do not be your own bottleneck

All the wire optimization in the world cannot save an endpoint whose handler is slow. Two server-side patterns deserve a place in any API performance conversation.

### The API N+1 query

The mirror image of the API-level chattiness from section 5 lives *inside* your handler: the **database N+1**. You fetch a list of orders in one query, then loop over them and fetch each order's customer in a separate query — N+1 database round-trips inside a single API request. The fix is the same shape as the API fix: batch the lookups into one query (a `JOIN` or a `WHERE id IN (...)`), backed by an index so the database does not scan. This is squarely a database concern, and the index that makes it cheap is the subject of [B-trees and how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) — the short version is that a covering index turns "fetch 50 customers by id" from 50 scans into one index-only lookup.

```python
# N+1: one query for orders, then one per order for the customer. Avoid.
orders = db.query("SELECT * FROM orders WHERE user_id = ? LIMIT 50", user_id)
for o in orders:
    o.customer = db.query("SELECT * FROM customers WHERE id = ?", o.customer_id)

# Batched: two queries total, regardless of how many orders. Prefer.
orders = db.query("SELECT * FROM orders WHERE user_id = ? LIMIT 50", user_id)
ids = {o.customer_id for o in orders}
customers = db.query("SELECT * FROM customers WHERE id IN (?)", ids)  # one round-trip
by_id = {c.id: c for c in customers}
for o in orders:
    o.customer = by_id[o.customer_id]
```

### Async I/O and not blocking the event loop

An API server spends most of its time *waiting* — for the database, for a downstream service, for the disk. A synchronous, thread-per-request model ties up a whole thread (and its memory) for each in-flight request, so a slow dependency exhausts your thread pool and the server stops accepting work even though the CPU is idle. **Async I/O** lets one thread juggle thousands of in-flight requests by parking each one while it waits and resuming it when its I/O completes. The trap is a single blocking call — a synchronous database driver, a CPU-bound JSON serialization, a `time.sleep` — that stalls the *entire* event loop and freezes every concurrent request. The discipline is: keep the event loop for I/O, push CPU-bound work to a thread or process pool. The language-level mechanics of this — event loops, the cost of blocking, where the GIL bites — are the domain of [why Python is slow and what fast actually means](/blog/software-development/python-performance/why-python-is-slow-and-what-fast-actually-means) and the async-in-practice posts in that series.

### Caching: the fastest request is the one you never make

The cheapest response is the one you can answer without doing any work. HTTP caching — `ETag`, `Cache-Control`, conditional requests — lets a client (or a CDN, or a gateway) skip both the server work and most of the transfer for unchanged resources. A conditional `GET` with `If-None-Match` that matches the current `ETag` returns a tiny `304 Not Modified` with no body: you pay one round-trip and a few bytes instead of regenerating and re-sending the whole payload.

```http
GET /v1/orders/ord_1a2b HTTP/2
If-None-Match: "v3-9c1f"
```

```http
HTTP/2 304 Not Modified
ETag: "v3-9c1f"
Cache-Control: private, max-age=30
```

That 304 is the payload lever (no body) and the server-time lever (no regeneration) at once, paid for with a single round-trip. This series devotes a whole post to it — [caching with ETags, Cache-Control, and conditional requests](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation) — and it belongs on every read-heavy endpoint.

## 9. Serialization cost: JSON parse versus protobuf

The last term in our latency equation is parse time — turning bytes into objects on each end. For small bodies it is negligible. For large ones, or for high-throughput service-to-service traffic where you do millions of calls a second, serialization becomes a real CPU and latency cost, and the format you chose at design time decides how much you pay.

JSON's strengths are exactly its costs. It is human-readable, schemaless, and universally supported — and it is *text*, so the parser must scan character by character, interpret escapes, parse numbers from decimal strings, and allocate a string for every key on every object. A binary format like **Protocol Buffers** (protobuf, the wire format behind gRPC) is the opposite trade: a schema-defined, length-prefixed binary encoding that is smaller on the wire and far cheaper to parse, because field tags are integers and values are already in their binary form — no character scanning, no number-from-string parsing.

```protobuf
syntax = "proto3";

message Order {
  string id = 1;
  string status = 2;
  string total = 3;        // money as a decimal string, never a float
  int64 created_at = 4;    // unix millis
}

message OrderList {
  repeated Order data = 1;
  string next_cursor = 2;
}
```

There is a subtlety even within "use JSON": *how* you parse it matters as much as the format. A streaming or SAX-style JSON parser that pulls fields you care about without materializing the whole document can be far cheaper than a parse-into-a-giant-object-then-throw-most-away approach. Likewise, on the server, a serializer that writes directly to the response stream beats one that builds a complete string in memory first. These are constant-factor wins, but on a hot path doing millions of calls a second, the constant factor is real money — and the language-level details of *why* a Python `json.loads` of a large body is slow, and what to do about it, are the domain of the [Python performance series](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o), which is where to go when serialization genuinely shows up as your bottleneck.

The honest framing: protobuf typically produces bodies a few times smaller than equivalent JSON and parses several times faster, with the gap widening for large or deeply nested messages. But you pay in **developer experience** — you cannot `curl` and read a protobuf body, you need codegen and a shared schema, and it is awkward for browser clients. So the rule is: **JSON for external, browser-facing, and low-volume APIs where DX wins; protobuf/gRPC for high-volume internal service-to-service traffic where the serialization cost actually shows up in your profile.** This series compares the paradigms in [gRPC and Protocol Buffers](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming) and [choosing a paradigm by force](/blog/software-development/api-design/choosing-a-paradigm-rest-vs-grpc-vs-graphql-by-force); the performance point here is narrow and important — **do not reach for protobuf because it is faster in a microbenchmark; reach for it because your profiler shows serialization is your bottleneck.** Most external APIs are bottlenecked on the network, not the parser, and there JSON-plus-gzip is the right call. The discipline of measuring before optimizing, and of knowing that the biggest speedups come from changing the algorithm (here, the round-trip count) not the constant factor (here, the parser), is exactly the lesson of [algorithmic complexity: the biggest speedups come from Big-O](/blog/software-development/python-performance/algorithmic-complexity-the-biggest-speedups-come-from-big-o).

## 10. Streaming and pagination for big results

When a result set is genuinely large — a year-end export, a full transaction history — neither a fatter page nor a faster serializer is the answer. Holding a million rows in memory to serialize them into one giant JSON array is slow to start (the client waits for the *whole* thing before it sees anything), memory-hungry on both ends, and fragile (one timeout loses all of it). Two patterns fix it.

**Pagination** breaks the set into bounded chunks the client pulls one at a time. For a stable, scalable cursor over a table that is being written to while you page, you want keyset/cursor pagination, not offset — offset re-scans and *skips or duplicates rows* when the table shifts under you. The full treatment is in [pagination: offset, cursor, and keyset](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale); the performance angle is that a bounded page is a bounded payload and a bounded server cost, every time.

**Streaming** sends the rows as they are produced so the client can start consuming before the server finishes generating, and neither side holds the whole set in memory. Newline-delimited JSON (NDJSON) is a simple, widely supported streaming format — one JSON object per line — that you serve with a chunked transfer encoding:

```http
GET /v1/orders/export HTTP/2
Accept: application/x-ndjson
```

```http
HTTP/2 200 OK
Content-Type: application/x-ndjson
Transfer-Encoding: chunked

{"id":"ord_1a2b","status":"shipped","total":"49.99"}
{"id":"ord_3c4d","status":"pending","total":"129.00"}
```

The server writes each line as it reads each row from the database cursor, flushes, and moves on; memory stays flat regardless of result size, and the client's time-to-first-row drops from "after the whole export" to "almost immediately." For push-style streams (live order updates) the API choices are Server-Sent Events, WebSocket, and gRPC streaming, which this series covers in [streaming APIs](/blog/software-development/api-design/streaming-apis-sse-websockets-and-server-streaming). The design rule: **bound the unbounded.** Never return a result set whose size the caller cannot control or predict; paginate it or stream it.

## 11. The latency-lever ladder: which lever, in what order

Put it all together as a decision ladder. When an endpoint is slow, you do not start at the bottom and tune the serializer; you start at the top, because the top levers remove *floors* that the bottom levers can never reach. But — and this is the whole discipline — you only know which rung you are on by **measuring first.**

![A vertical stack of the latency-lever ladder, ordered from cutting round-trips and taming the tail at the top down through shrinking the payload, compressing, caching, and faster parsing](/imgs/blogs/api-performance-payload-size-compression-and-tail-latency-8.png)

| Lever | Typical win | When it is the bottleneck |
| --- | --- | --- |
| Cut round-trips (batch, reuse connection) | removes whole multiples of RTT | chatty client, N+1 at the API, fresh connection per call |
| Tame the tail (hedge, timeout, breaker) | collapses p99 toward p50 | a fan-out where one slow dependency owns the response |
| Shrink payload (project, paginate) | 2x to 5x transfer | over-fetching, unbounded pages, fat denormalized bodies |
| Compress the wire (gzip, brotli) | 5x to 10x on JSON | large text bodies on a thin link |
| Cache (ETag, Cache-Control, CDN) | a 304 or a cache hit, near zero | repeated reads of slowly changing resources |
| Faster parse (protobuf, streaming) | a few times on hot paths | profiler shows serialization is the cost |

Read the table top to bottom. The two top rungs are *structural* — they change how many times you cross the network and how the system behaves under a slow component — and they deliver the largest, most durable wins. The middle rungs are *byte-level* — payload and compression — and deliver large wins on thin links. The bottom rungs are *constant-factor* — caching and serializer choice — and matter once the structure is right. **The mistake is to start at the bottom**: swapping JSON for protobuf to fix an endpoint whose real problem is eleven round-trips is a week of work that moves the number by single-digit milliseconds while the 1.5-second floor stays.

This is the API-design face of the optimization loop: profile to find the bottleneck, fix the bottleneck, re-measure, repeat. The general version of that loop — latency numbers every engineer should know, and the discipline of measuring before optimizing — is the subject of [a mental model of performance](/blog/software-development/python-performance/a-mental-model-of-performance-latency-numbers-and-the-optimization-loop), and the observability tooling that finds the bottleneck for an API is in [observability for APIs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos).

## Case studies and real-world references

These are well-documented, and where I cite a number I frame it as a published claim or a range, not a benchmark I ran.

**Google: "The Tail at Scale."** Jeffrey Dean and Luiz André Barroso's 2013 Communications of the ACM article is the canonical text on tail latency. Its central argument is the probability we derived: in a system where a user request fans out to many servers, the overall latency is governed by the slowest component, so even rare per-server slowness becomes common at the request level. The paper's signature mitigation is **hedged and tied requests** — sending a backup request after a brief delay and taking the first response — which the authors report can dramatically reduce the tail (their data showed large p99 reductions for a small percentage of extra requests). It is the source for the hedging, deadline, and tail-tolerance techniques in section 7.

**Brotli adoption.** Brotli was developed at Google and standardized in RFC 7932; it is supported by all major browsers and widely available in CDNs and web servers. The published and widely reproduced finding is that brotli compresses text (HTML, CSS, JavaScript, JSON) meaningfully smaller than gzip at comparable settings, with the largest gains at its highest quality levels — which is exactly why it is preferred for *static, cacheable* assets where you compress once. For dynamic responses, the high-quality brotli levels are too CPU-expensive on the hot path, so gzip or low-level brotli remains the pragmatic choice. The honest framing: the ratio advantage is real and modest for text; the right use depends on cacheability.

**HTTP/2 and HTTP/3 gains.** HTTP/2 is specified in RFC 9113 and HTTP/3 in RFC 9114. The well-established result is that HTTP/2's multiplexing eliminates the application-layer head-of-line blocking and per-connection limits of HTTP/1.1, and HTTP/3 over QUIC additionally removes the TCP-level head-of-line blocking that still affects HTTP/2 on lossy links, while cutting handshake round-trips. The size of the real-world improvement is workload-dependent — biggest on high-latency, lossy mobile links and small on clean wired connections — so I frame it as "measurable on mobile, marginal on a fast LAN," not a fixed percentage.

**Mobile payload wins.** Multiple engineering teams have published that the dominant cost for mobile API calls on poor links is transfer and round-trips, not server time, and that trimming payloads and reducing call counts produced the largest user-visible improvements. This matches the math in sections 2 and 5 and the incident that opened this post. The general lesson — that the bottleneck for a remote caller is almost never your handler's CPU — is the most reliable finding in the area.

## When to reach for this (and when not to)

Performance work is a trade against complexity and developer time, so be decisive about when *not* to spend it.

- **Do measure before optimizing.** If you have not looked at the p99 and a trace of where the time goes, you are guessing, and guesses about performance are wrong more often than right. Reach for the profiler and the trace first, always.
- **Do turn on compression by default** for any text response over about 1 KB, with `Vary: Accept-Encoding`. It is nearly free and one of the highest-leverage changes you can make. The exception: skip already-compressed content types and tiny bodies.
- **Do cut round-trips first** when a client is chatty — batch endpoints, `include`, a BFF, or GraphQL. This is the top of the ladder and the most durable win.
- **Do not reach for protobuf/gRPC for an external, browser-facing, or low-volume API** just because it is "faster." You lose `curl`-ability, you need codegen and a shared schema, and the serialization cost you are saving is invisible next to the network. JSON-plus-gzip is the right default; switch only when your profiler says serialization is the bottleneck.
- **Do not hedge non-idempotent requests.** A hedged `POST` without an idempotency key is a double-charge waiting to happen. Hedge only safe or idempotent operations, and always cancel the loser.
- **Do not retry without a budget and backoff.** Naive retries turn a slow dependency into a dead one via a retry storm. Cap retries as a fraction of traffic and use exponential backoff with jitter.
- **Do not return an unbounded result set.** Paginate it or stream it. An endpoint whose payload size the caller cannot bound is an endpoint that will eventually time out and serialize-blow your memory.
- **Do not optimize the fast dependencies in a fan-out.** Find the one that owns the tail and fix that. Shaving the p99 of a service that is not the long pole moves nothing.
- **Do not micro-optimize what the network dominates.** On a remote link, server time is often a rounding error. Five milliseconds saved in your handler is invisible behind 150 ms of RTT and 300 ms of transfer.

## Key takeaways

- **There is no single "the latency."** It is a distribution across callers, networks, and request shapes; optimize the bottleneck for the caller who matters, which on a remote link is rarely your handler's CPU.
- **Transfer time is roughly $\text{size}/\text{bandwidth} + \text{RTT}$, and round-trips are a floor you cannot compress away.** Cutting $n_{\text{rtt}}$ — batch, reuse connections, multiplex — beats every byte-level lever.
- **Stop sending bytes nobody reads.** Sparse fieldsets, projection, and bounded pages cut payloads multiplicatively before you ever compress.
- **Compress text by default.** gzip shrinks JSON 5 to 8 times for low CPU; brotli wins on cacheable assets; set `Vary: Accept-Encoding`, never double-compress, and skip tiny or already-compressed bodies.
- **Reuse connections.** A reused connection amortizes the 300-to-450 ms cold handshake to near zero; HTTP/2 multiplexes, HTTP/3 over QUIC removes the last head-of-line floor on lossy links.
- **The p99, not the average, is what your users feel** — and a fan-out to $n$ dependencies has latency equal to the *maximum*, so $P(\text{all fast}) = p^{n}$ shrinks fast and the slowest dependency owns your tail.
- **Tolerate the tail with hedged requests, propagated deadlines, retry budgets, circuit breakers, and load shedding** — and only hedge idempotent operations.
- **Measure first, fix the bottleneck, re-measure.** Work the latency-lever ladder from the top (structural) down (constant-factor); never start by swapping the serializer.

## Further reading

- Jeffrey Dean and Luiz André Barroso, "The Tail at Scale," *Communications of the ACM*, 2013 — the canonical paper on tail latency, fan-out amplification, and hedged requests.
- RFC 9113, *HTTP/2* — the multiplexing and stream model that kills application-layer head-of-line blocking.
- RFC 9114, *HTTP/3* — HTTP over QUIC, with per-stream loss recovery and faster handshakes.
- RFC 9110, *HTTP Semantics* — the definitive reference for methods, status codes, and the `Accept-Encoding`/`Content-Encoding` content-negotiation machinery.
- RFC 7932, *Brotli Compressed Data Format* — the brotli specification.
- Ilya Grigorik, *High Performance Browser Networking* (O'Reilly) — DNS, TCP, TLS, HTTP/2, and the network-level latency model in depth; freely readable online.
- Within this series: the intro hub [what is an API, the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the [pagination](/blog/software-development/api-design/pagination-offset-cursor-and-keyset-tradeoffs-at-scale) and [filtering and sparse fieldsets](/blog/software-development/api-design/filtering-sorting-and-sparse-fieldsets-without-reinventing-sql) posts (payload levers); [caching with ETags](/blog/software-development/api-design/caching-etags-cache-control-conditional-requests-invalidation); [observability for APIs](/blog/software-development/api-design/observability-for-apis-logs-metrics-traces-and-slos) (measure first); and the capstone [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
