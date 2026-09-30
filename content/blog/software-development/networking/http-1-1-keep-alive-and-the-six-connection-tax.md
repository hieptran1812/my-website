---
title: "HTTP/1.1 Keep-Alive and the Six-Connection Tax"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Learn to separate cold connection cost, browser socket queues, HTTP framing, and deliberate connection closure with measurements you can reproduce."
tags:
  [
    "networking",
    "distributed-systems",
    "http-1-1",
    "keep-alive",
    "connection-pooling",
    "browser-performance",
    "latency",
    "chunked-transfer",
    "pipelining",
  ]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 43
image: "/imgs/blogs/http-1-1-keep-alive-and-the-six-connection-tax-1.webp"
---

A page asks for twelve small resources. The server finishes each one quickly, yet the browser waterfall has long blank stretches. Another client calls the same API repeatedly and reports that the first request is slow while the rest are fast. A third client intermittently pays the first-request price again. These can be three manifestations of connection management, but the fixes differ. The opening path map below locates the setup steps: resolution finds the address, TCP and TLS establish a usable path, and the HTTP exchange follows. The reused socket follows the lower row and bypasses new setup. HTTP/1.1 keep-alive changes whether the next request pays the TCP and TLS portions again. It does not make one HTTP/1.1 connection capable of carrying independently interleaved responses.

![Path map comparing cold HTTP setup with a reused socket](/imgs/blogs/http-1-1-keep-alive-and-the-six-connection-tax-1.webp)

If you are starting from the whole path, read [what actually happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). This post follows one connection after name resolution and before application policy. [The TCP handshake and what it costs you](/blog/software-development/networking/the-tcp-handshake-and-what-it-costs-you) owns the SYN exchange. [TLS and its handshake cost](/blog/software-development/networking/tls-what-the-handshake-buys-and-what-it-costs) owns authentication and key establishment. Here we account for how often the application forces those costs to recur.

The operational rule is simple: count *connections created*, *requests completed per connection*, and *time spent waiting for a usable connection*. A request trace that contains only server time cannot settle any of those questions. Neither can a single ping. The rest of the post turns that rule into protocol mechanics, arithmetic, and a lab.

## 1. A connection is a reusable lane, not a free lane

HTTP/1.1 runs over a transport connection. For a conventional HTTPS origin, a cold request may need DNS, a TCP three-way handshake, a TLS handshake, a request transmission, server work, and a response. A warm request can reuse an open transport and skip the connection setup. This is why a pool of idle sockets is useful. It is also why idle sockets consume file descriptors, memory, proxy state, and sometimes backend admission slots. Reuse buys latency by reserving resources.

The standard's behavior matters here. [RFC 9112, published June 2022, section 9.3](https://www.rfc-editor.org/rfc/rfc9112.html#section-9.3) says HTTP/1.1 connections are persistent by default. Neither side needs to send `Connection: keep-alive` to make that true. The `Connection: close` option says the connection will not persist after the current response. An HTTP/1.0 peer follows different rules, so check the response version when a legacy hop is involved. The `Connection` field is hop by hop. A browser can reuse its connection to an edge proxy while that proxy opens a fresh one to the origin, or the reverse. A client-side trace only proves reuse on the client-facing hop.

Persistence has a strict precondition: both peers must know exactly where one message ends before interpreting bytes as the next message. A `Content-Length`, a complete chunked body, or a response status whose rules imply no body can provide that boundary. A body whose end is defined only by connection closure cannot leave a connection available for another response. This is a framing rule, not a tuning preference. RFC 9112 also requires a server to read the entire request body or close after responding; otherwise unread body bytes could be parsed as the next request. The client likewise must consume the full response body before reuse. The distinction becomes crucial when a program reads headers, abandons the body, and wonders why its pool keeps dialing.

### Price a cold request without pretending every path is identical

Consider a *modeled* path with measured round-trip time $R=80\,\mathrm{ms}$, a fresh TCP connection, a full TLS 1.3 handshake, no DNS miss, negligible serialization for a small request, and `20 ms` of server work. The TCP handshake costs approximately one RTT before a conventional client can send application data. A full TLS 1.3 handshake usually adds another RTT before ordinary application data, subject to implementation and transport details. The request and first response byte then take approximately one further RTT plus server work. Our explanatory first-byte model is

$$
T_{\mathrm{cold}} \approx R_{\mathrm{TCP}} + R_{\mathrm{TLS}} + R_{\mathrm{request/response}} + T_{\mathrm{server}}
= 80 + 80 + 80 + 20 = 260\,\mathrm{ms}.
$$

For a warm connection under the same assumptions, the model is $T_{\mathrm{warm}} \approx R+T_{\mathrm{server}} = 100\,\mathrm{ms}$. The derived difference is $160\,\mathrm{ms}$ per unnecessary cold establishment. This is not a production benchmark. TLS resumption, 0-RTT, a proxy, coalescing, packet loss, cached DNS, CPU load, and actual server processing change the value. The TCP and TLS round-trip structure comes from [RFC 9293, August 2022](https://www.rfc-editor.org/rfc/rfc9293.html) and [RFC 8446, August 2018](https://www.rfc-editor.org/rfc/rfc8446.html). The numbers come solely from the stated scenario.

If a page independently opens six cold HTTPS connections on this modeled path, they can handshake in parallel. You cannot add $6\times160\,\mathrm{ms}$ to page wall-clock time as though the handshakes were serialized. The aggregate extra setup work is real, but the visible delay depends on scheduling, dependencies, CPU, congestion, and what the browser already has open. This distinction prevents a common arithmetic mistake: per-connection work is not necessarily critical-path latency.

### What the timings in a client actually measure

`curl -w` exposes `time_namelookup`, `time_connect`, `time_appconnect`, `time_pretransfer`, `time_starttransfer`, and `time_total`. These are cumulative points on a request timeline, not independent durations. On a cold TLS request, `time_connect - time_namelookup` approximates time from resolution to established TCP connection; `time_appconnect - time_connect` covers the TLS stage; `time_starttransfer - time_pretransfer` includes request flight, server work, and response flight. On a reused connection, the setup fields may be zero or otherwise reflect reuse semantics rather than another handshake. The counter `num_connects` and verbose line `Re-using existing connection` are better evidence of reuse than one unusually fast response. Check your installed curl's `--write-out` documentation for field availability and interpretation; [curl's own write-out reference](https://curl.se/docs/manpage.html#-w) is the primary source.

The measurements tell us which *stage* enlarged. They do not by themselves identify which *hop*. A load balancer can maintain one browser connection and another origin connection. For the full path, pair client timing with edge and server connection counters, or with a narrowly filtered capture. [The senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) collects that cross-layer method.

## 2. Keep-alive does not mean multiplexing

Imagine a checkout lane. Keeping the lane open saves the work of building a new lane for the next customer. It does not allow two customers to occupy the scanner at once. Ordinary HTTP/1.1 clients send a request and await its response on a socket before sending the next request on that socket. Multiple sockets create multiple lanes. This is how a browser can download several independent resources concurrently despite each HTTP/1.1 connection's serial response stream.

![Graph comparing new connections, sequential reuse, and parallel HTTP/1.1 sockets](/imgs/blogs/http-1-1-keep-alive-and-the-six-connection-tax-2.webp)

There is a protocol nuance. HTTP/1.1 permits *pipelining*: a client may send more than one request on a persistent connection without waiting for earlier responses. The server may even process safe requests in parallel. But [RFC 9112 section 9.3.2](https://www.rfc-editor.org/rfc/rfc9112.html#section-9.3.2) requires corresponding responses in request order. There is no stream identifier in HTTP/1.1 that would let the client independently assemble response B ahead of response A on the same connection. This is why pipelining is not multiplexing. The [HTTP/2 post](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer) follows the framing change that introduced concurrent streams, and the [HTTP/3 post](/blog/software-development/networking/quic-and-http-3-what-moving-to-udp-actually-changed) follows what happens when transport blocking moves again.

Suppose a slow response A needs $400\,\mathrm{ms}$ of server work and a quick response B needs $10\,\mathrm{ms}$. These are *chosen values in a model*, not observed timings. On a pipelined HTTP/1.1 socket with A sent first, B's response cannot be delivered ahead of A's response. Even if B finishes its server computation early, the wire order keeps B behind A. With two independent sockets, B need not wait for A's response framing. With HTTP/2 streams, B can use its own response frames, although loss of one TCP segment can still stall both streams at the transport layer. The point is causal: protocol concurrency and transport concurrency are different.

### Reuse can turn into a queue

An application pool usually has a maximum number of open connections per destination. If every connection is busy, another request waits for a connection. A browser has its own socket scheduler and per-origin limits for HTTP/1.x. A reverse proxy may maintain another queue and another pool behind it. If the server answers in $10\,\mathrm{ms}$ but the client reports $210\,\mathrm{ms}$, do not call the difference server latency until you locate socket acquisition. A connection pool can keep tail latency invisible to backend traces.

Take a deliberately simple queue model: twelve equally sized resources become ready at once; one origin allows six active HTTP/1.1 connections; each resource occupies a connection for $100\,\mathrm{ms}$ after establishment; connections are already warm; and there are no priority or bandwidth interactions. The first six complete after roughly $100\,\mathrm{ms}$. The second six start only as lanes free, and finish after roughly $200\,\mathrm{ms}$. The extra $100\,\mathrm{ms}$ comes from client-side scheduling, even though per-resource server behavior is identical. The model deliberately omits TCP fairness, browser priorities, cache hits, and transfer dependencies. It demonstrates the queue, not a universal page-load prediction.

The animated figure makes that queue visible. Six lanes fill, queued requests wait, then requests enter lanes as responses complete. The important motion is the handoff from waiting to active. Freeze that motion and it is easy to mistake the queued work for slow server processing.

<figure class="blog-anim">
<svg viewBox="0 0 860 320" role="img" aria-label="Six browser sockets serve six HTTP/1.1 requests while requests seven and eight wait, then take freed sockets" style="width:100%;height:auto;max-width:860px">
<style>
.h11-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.h11-active{fill:var(--accent,#6366f1)}.h11-wait{fill:#eab308}.h11-text{font:600 17px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.h11-small{font:600 14px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.h11-white{font:700 16px ui-sans-serif,system-ui;fill:white;text-anchor:middle}
@keyframes h11-first{0%,30%{opacity:1}40%,88%{opacity:0}100%{opacity:1}}
@keyframes h11-replace{0%,34%{transform:translate(0,0)}48%,88%{transform:translate(0,-126px)}100%{transform:translate(0,0)}}
@keyframes h11-status{0%,34%{opacity:0}46%,88%{opacity:1}100%{opacity:0}}
.h11-first{animation:h11-first 12s ease-in-out infinite}.h11-replace{animation:h11-replace 12s ease-in-out infinite}.h11-status{animation:h11-status 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.h11-first,.h11-replace,.h11-status{animation:none}.h11-first{opacity:0}.h11-status{opacity:1}.h11-replace{transform:translate(0,-126px)}}
</style>
<text class="h11-text" x="20" y="36">One origin: six browser socket slots</text>
<rect class="h11-bg" x="20" y="56" width="125" height="126" rx="12"/><rect class="h11-bg" x="158" y="56" width="125" height="126" rx="12"/><rect class="h11-bg" x="296" y="56" width="125" height="126" rx="12"/><rect class="h11-bg" x="434" y="56" width="125" height="126" rx="12"/><rect class="h11-bg" x="572" y="56" width="125" height="126" rx="12"/><rect class="h11-bg" x="710" y="56" width="125" height="126" rx="12"/>
<text class="h11-small" x="45" y="84">Socket 1</text><text class="h11-small" x="183" y="84">Socket 2</text><text class="h11-small" x="321" y="84">Socket 3</text><text class="h11-small" x="459" y="84">Socket 4</text><text class="h11-small" x="597" y="84">Socket 5</text><text class="h11-small" x="735" y="84">Socket 6</text>
<rect class="h11-active" x="42" y="106" width="82" height="52" rx="8"/><rect class="h11-active h11-first" x="180" y="106" width="82" height="52" rx="8"/><rect class="h11-active" x="318" y="106" width="82" height="52" rx="8"/><rect class="h11-active" x="456" y="106" width="82" height="52" rx="8"/><rect class="h11-active h11-first" x="594" y="106" width="82" height="52" rx="8"/><rect class="h11-active" x="732" y="106" width="82" height="52" rx="8"/>
<text class="h11-white" x="83" y="138">R1</text><text class="h11-white h11-first" x="221" y="138">R2</text><text class="h11-white" x="359" y="138">R3</text><text class="h11-white" x="497" y="138">R4</text><text class="h11-white h11-first" x="635" y="138">R5</text><text class="h11-white" x="773" y="138">R6</text>
<text class="h11-text" x="20" y="220">Waiting for a free socket</text>
<g class="h11-replace"><rect class="h11-wait" x="180" y="232" width="82" height="52" rx="8"/><text class="h11-white" x="221" y="265">R7</text></g><g class="h11-replace"><rect class="h11-wait" x="594" y="232" width="82" height="52" rx="8"/><text class="h11-white" x="635" y="265">R8</text></g>
<text class="h11-small h11-status" x="285" y="312">R2 and R5 finish; R7 and R8 reuse those sockets</text>
</svg>
<figcaption>With six origin sockets busy, requests seven and eight wait. Each advances only when a socket becomes free; six is a browser policy here, not an HTTP RFC limit.</figcaption>
</figure>

The useful diagnostic question is not simply “how many requests are pending?” It is “where were they pending?” Browser devtools can display a queueing or stalled phase, but its exact accounting varies with browser version. A server trace usually begins after the request reaches the server, so it excludes browser queue wait. At the API client, instrument pool-acquisition duration separately from connect, TLS, write, first byte, and body read. If your chosen client exposes no acquisition hook, measure the number of concurrent requests and open sockets together and use a packet capture to establish when each request actually went onto the wire.

### Separate occupancy from connection count

A pool of six connections can be healthy or saturated. Six established sockets tell us capacity that *exists*, not capacity that is *available*. If five sockets are idle and one response is streaming slowly, a new small request has a lane. If all six are occupied by long responses, it waits, even if the server could compute the answer immediately. A browser waterfall's queue phase and a service client's pool-acquire timer describe this waiting. The server cannot log a request it has not received, so a dashboard built only from server spans systematically hides it.

When investigating, sample three time series together: active connections, idle connections, and waiters. A rising waiter count while active connections stay at the configured maximum suggests a client-side capacity limit. A rising connection creation rate with low waiters suggests frequent churn or cold sockets, a different problem. A rising established count with a stable request rate may simply mean idle sockets are retained longer. These are directional interpretations, not universal thresholds. Check whether the metrics refer to the browser, edge, or origin pool before comparing them.

Response size changes occupancy. Suppose a connection spends $20\,\mathrm{ms}$ waiting for server computation and $80\,\mathrm{ms}$ transferring a body in a modeled scenario. Its total occupancy is roughly $100\,\mathrm{ms}$. Reducing server computation from $20$ to $10\,\mathrm{ms}$ only saves $10\,\mathrm{ms}$ of slot time; a transfer bottleneck still dominates. By contrast, cutting an unnecessary $80\,\mathrm{ms}$ body transfer, perhaps through a cache hit or a smaller representation, can free the slot much earlier. That does not mean compression is always the answer: compression can consume CPU and alter first-byte time. It means pool design has to account for how long a response monopolizes a connection, not only how long the handler computes.

A streamed response makes this especially visible. If a server sends one event every few seconds over a long-lived HTTP/1.1 response, that socket remains occupied for the stream's lifetime. The client cannot simply return it to the pool after reading the first event. If several streams share the same origin and the browser is using HTTP/1.1, the remaining short requests may run out of lanes. The [WebSockets and SSE post](/blog/software-development/networking/websockets-sse-and-long-lived-connections) treats the full lifecycle and proxy behavior. Here the relevant fact is that an unfinished response holds a serial HTTP/1.1 lane.

### Why opening more sockets can backfire

An extra socket creates another independent response lane, but it also creates more transport and server state. On a constrained access link, extra flows compete for the same bandwidth and queue. If the bottleneck is total link capacity, twelve sockets cannot transmit twice the useful bytes of six without changing the link. They can alter fairness, packet loss, and scheduling enough to make small-resource latency better or worse. That is why the sharding calculation later compares critical paths rather than equating socket count with throughput.

A proxy may also enforce a maximum concurrent connection count, per-client admission policy, or file-descriptor ceiling. Browser sharding can present more connections to that proxy even when every hostname resolves to the same service. A defensive control may then treat the traffic as abusive. MDN notes that pushing parallel connections beyond a common browser limit can trigger server-side DoS protection. The observation is a warning about deployment interaction, not an argument for a universal cap of six on every server.

Finally, connection reuse can improve more than the TCP/TLS setup line in a waterfall. A long-lived TCP connection can retain congestion-control state and avoid repeatedly restarting from a cold state, while a completely new connection begins with new transport state. The magnitude depends on congestion control, path, transfer size, and idle behavior. For short API responses on a fast local network, setup RTT may dominate. For larger transfers over a long path, slow start and loss history may matter. [Congestion control, CUBIC, and BBR](/blog/software-development/networking/congestion-control-cubic-bbr-and-what-changing-it-actually-does) owns that transport behavior; do not assign all warm-connection benefit to one handshake number.

## 3. The six-connection tax is a browser policy

There is no HTTP/1.1 specification line that says a browser must use six connections. The frequently quoted “six per origin” is a common browser practice, described by [MDN's HTTP/1.x connection management guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Connection_management_in_HTTP_1.x). Real limits can vary by browser, version, proxy mode, credentials, connection state, or scheduling policy. Treat six as an explicit scenario assumption until you verify it in the browser under test.

One origin usually means a particular scheme, host, and port in web security terminology. Connection pooling keys can have more dimensions, such as proxy configuration, privacy partition, certificate properties, and browser implementation policy. We should therefore avoid promoting a pedagogical limit into a portable guarantee. The symptom to look for is a roughly flat maximum of active HTTP/1.1 sockets to the same destination while requests sit queued in the client.

The old workaround was domain sharding: put assets on multiple hostnames that resolve to the same service. A browser that budgets six HTTP/1.1 sockets for each hostname could open more total sockets. This was a rational response to serial response streams and a local cap. It could also increase DNS work, TCP and TLS handshakes, server socket counts, and congestion competition. MDN explicitly warns that sharding is generally a bad fit for HTTP/2, where concurrent streams and possible connection coalescing change the calculation. A sharded URL list can survive long after the protocol migration that made it counterproductive.

![Before and after model of one origin versus sharded HTTP/1.1 hostnames](/imgs/blogs/http-1-1-keep-alive-and-the-six-connection-tax-4.webp)

Here is the derived arithmetic for the same twelve ready resources. Suppose six warm sockets per hostname, each resource holds one socket for $100\,\mathrm{ms}$, and a second hostname has *no* warm sockets. On one hostname, two $100\,\mathrm{ms}$ waves take about $200\,\mathrm{ms}$ from first request dispatch. Split six resources onto each of two hostnames and the transfer phase could collapse to one $100\,\mathrm{ms}$ wave. But if the extra hostname costs $80\,\mathrm{ms}$ TCP plus $80\,\mathrm{ms}$ TLS setup on the critical path, its six resources may finish around $260\,\mathrm{ms}$, slower than the unsharded $200\,\mathrm{ms}$ scenario. This is a model with hand-picked values, not a benchmark. It exposes the crossover condition:

$$
T_{\mathrm{extra\ setup}} \lt T_{\mathrm{queue\ wave\ avoided}}
$$

where both sides refer to the additional hostname's critical path under the same workload. A DNS miss, bandwidth competition, or an extra certificate validation cost moves the crossover. A warm existing connection to the second hostname moves it the other way. Do not use a blanket “sharding halves load time” rule.

| Scenario assumption | Active sockets available | Transfer waves for twelve equal objects | Extra cold setup on critical path | Source |
| --- | ---: | ---: | ---: | --- |
| One warm hostname, six-socket policy | 6 | 2 | 0 ms | Derived here: ceiling of 12 divided by 6 |
| Two warm hostnames, six each | 12 | 1 | 0 ms | Derived here: ceiling of 12 divided by 12 |
| Second hostname starts cold | Up to 12 after setup | 1 | Modeled 160 ms | Derived here from the 80 ms RTT TCP plus TLS assumptions |

The table separates concurrency from setup. A well-designed experiment would also hold cache state, response sizes, browser version, network path, and server capacity fixed. If a CDN serves two hostnames from different edges, the experiment no longer isolates sharding.

### Use the right denominator for a sharding experiment

A page may issue many more requests than it opens sockets. Request count is a poor denominator for judging reuse unless we also know concurrency. Consider sixty objects and six sockets. If the objects are small and discovered in ten sequential waves, each socket may serve about ten requests; a rough observed ratio of ten requests per connection could be excellent. If those sixty objects are all revealed at once and each holds a socket for a long time, the same total request and connection counts can hide an enormous queue. We need request start time, completion time, socket identity, and queue delay to distinguish the situations.

A second hostname can make this ratio look worse while improving one selected page metric. Two pools with six sockets each spread the same sixty requests across twice as many connections. Requests per connection can fall, but the earliest visible images might complete sooner because they had more simultaneous lanes. Conversely, if the extra hostname is cold, a dashboard may report better aggregate throughput after all handshakes complete while the user sees a slower first meaningful image. Always choose the user-visible metric first: time to a critical image, completion of all resources, or interaction readiness. Then explain it with socket and queue measurements. Optimizing the ratio itself is not the goal.

The browser's scheduler adds another confounder. It can prioritize stylesheets, scripts, images, or requests initiated by a service worker. A naive six-by-six wave picture assumes all work has equal priority and becomes ready at once. Real dependency graphs do not. A stylesheet might block discovery of a later resource, and a large low-priority image might start only when a lane is spare. That is why the animated figure should be read as a mechanism demonstration, not as a promise that every page uses exact synchronized waves. Browser devtools supply the actual request initiation order and priority for the target version.

When testing a sharded legacy page, collect a cold-cache run and a warm-cache run separately. Also distinguish cold *connection* from cold *HTTP cache*. A warm cache can avoid network requests entirely, whereas a warm connection still sends a request but skips setup. A CDN cache hit can avoid origin work while leaving browser-to-edge transfer unchanged. If all three are called “warm,” the experiment cannot identify which optimization helped. Use the browser's cache toggle, clear or preserve sockets intentionally, and record the negotiated protocol and DNS state for each trial. Repeat runs because connection establishment and browser scheduling have variance; do not report one best waterfall as the result.

## 4. Why HTTP/1.1 pipelining did not save the browser

Pipelining looked like the obvious way to remove a round trip between requests without opening more sockets. The client can write GET A, GET B, and GET C together. A compatible server has the requests early and may do work before the prior response ends. The constraint is that responses still arrive in request order. A large or slow A holds B and C behind it on that socket. This is application-level head-of-line blocking. An intermediary that mishandles a pipeline can add ambiguous failures and retry risk.

The failure mode is especially unpleasant after an asynchronous close. Suppose the client writes three requests and receives only one complete response before the connection closes. It must decide which outstanding operations can be retried safely. Safe and idempotent method semantics matter, and the client cannot blindly replay side-effecting requests. RFC 9112's pipelining section contains explicit retry precautions for this case. This is a much harder client state machine than “send one request, read one response, return socket to pool.” HTTP/2's stream identifiers and framing address the response association problem, but HTTP/2 has its own flow-control and TCP loss behavior.

Mozilla provides a dated public case. The [Firefox 54 developer release notes](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Releases/54) state that HTTP/1 pipelining support was removed. [Mozilla bug 1340655](https://bugzilla.mozilla.org/show_bug.cgi?id=1340655), resolved for the release on June 13, 2017, records the removal work in the browser implementation. The observed event is a feature removal, not an outage. The relevant user-visible risk was erratic loading through intermediaries and head-of-line delay; [MDN's connection management guide](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Connection_management_in_HTTP_1.x) lists buggy proxies, implementation complexity, and head-of-line blocking as reasons modern browsers do not enable pipelining by default. The trigger for Mozilla's change was an engineering choice to retire the H1 pipeline code. The contributing condition was a feature whose theoretical round-trip benefit did not translate into robust, general browser behavior. The transferable guardrail is to verify response ordering and intermediary behavior before assuming that a protocol-permitted optimization is a deployable browser optimization. Do not cite this case as evidence that every server was broken or that a specific percentage of pages failed; Mozilla's sources do not establish either claim.

![Causal grid of HTTP/1.1 pipelining and Firefox 54 removal](/imgs/blogs/http-1-1-keep-alive-and-the-six-connection-tax-6.webp)

This case also clarifies what HTTP/1.1 still offers. Persistence succeeds because reusing a connection after a *complete* response is straightforward. Parallel sockets succeed because each socket has an independent ordered response stream. Pipelining tried to use one ordered stream for multiple outstanding responses. Each step offers different concurrency and failure semantics. Calling all three “keep-alive” blurs exactly the distinctions we need during debugging.

## 5. Message framing decides whether the lane can stay open

A persistent stream has no built-in marker that says “the current HTTP response ended here.” TCP provides ordered bytes, not response records. HTTP/1.1 has to frame each message. This is the connection between `Content-Length`, chunked transfer coding, and reuse. It also explains why a server can close a connection to delimit a response but cannot then keep that same connection for the next response.

![HTTP/1.1 body framing with Content-Length and chunked terminal marker](/imgs/blogs/http-1-1-keep-alive-and-the-six-connection-tax-5.webp)

With `Content-Length: 12`, the recipient knows to consume twelve body octets, then the next octet can start another HTTP message. The value counts octets of the message body as framed on that hop, not characters in a rendered string and not total TCP bytes. Headers do not count toward it. If a response carries content coding such as gzip, the length applies to the encoded representation as sent. If the connection ends before the declared count, [RFC 9112 section 6.3](https://www.rfc-editor.org/rfc/rfc9112.html#section-6.3) says the message is incomplete.

For a body whose final length is unknown when headers go out, `Transfer-Encoding: chunked` sends a hexadecimal chunk size, that many bytes, and a CRLF for each chunk. A zero-size final chunk marks the end, followed by optional trailers and a final CRLF. A minimal illustrative body can be written as `5\r\nhello\r\n0\r\n\r\n`; the five counts the five bytes in `hello`. The zero does not mean “a zero-length application record” that can be ignored. It is the framing terminator that lets the receiver finish the response and safely consider reuse. [RFC 9112 section 7.1](https://www.rfc-editor.org/rfc/rfc9112.html#section-7.1) defines the wire syntax.

Transfer coding is hop by hop. A proxy can decode a chunked upstream response and send a length-delimited downstream response, or vice versa, provided it obeys each hop's framing rules. A packet capture on only one side of the proxy therefore cannot prove the other side's transfer coding. Compression is another axis: `Content-Encoding: gzip` describes representation coding, while `Transfer-Encoding: chunked` describes message framing. Confusing those fields often leads to bogus “chunked compression” explanations. The [API performance post](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) owns compression and payload trade-offs; this post owns how a recipient knows a response is complete.

Two conflicting length signals deserve special care. RFC 9112 says `Transfer-Encoding` overrides `Content-Length` if both arrive and notes that the combination can indicate request smuggling or response splitting. An intermediary may reject it. Engineers should not normalize ambiguous requests casually, because different parsers can disagree about where one request ends and the next begins. In operational debugging, preserve the raw bytes and the proxy chain, and consult the security team's handling policy. This post is about diagnosis, not a recipe for generating ambiguous traffic.

### Decode a chunked response before blaming the transport

Consider this deliberately small, illustrative response body. It is wire syntax, with each visible `\r\n` denoting a carriage return followed by a line feed rather than four printable characters:

```http
HTTP/1.1 200 OK\r\n
Transfer-Encoding: chunked\r\n
Content-Type: text/plain\r\n
\r\n
5\r\n
hello\r\n
6\r\n
 world\r\n
0\r\n
\r\n
```

The receiver reads the status and headers, then parses hexadecimal `5`, reads five body octets, parses hexadecimal `6`, reads six more, and finally sees the zero chunk. The reconstructed body is `hello world`. The two data chunks do not have to align with TCP packets: TCP can split a chunk across segments or combine several chunks in one segment. A packet capture that shows a TCP packet boundary between two strings does not prove an HTTP chunk boundary. Parse the HTTP bytes, not the packet count.

The sender can choose chunk sizes for implementation reasons, and a proxy can reframe the body on the next hop. The recipient must also process the final empty line after the zero chunk, with trailers if present, before considering the message complete. Until that point the connection cannot be safely returned to the pool for another response. If a server sends data chunks but never sends the terminating zero chunk and does not close, a client can wait until its timeout even though the application already produced every intended byte. If it closes without a complete chunked message, the response is incomplete. Changing the keep-alive timeout merely changes when the symptom surfaces; it does not repair the frame.

This detail is useful in an incident because a first-byte metric can look excellent while total time is terrible. The first chunk arrives promptly, the handler trace ends, and the client remains blocked waiting for the terminal frame. Check `Transfer-Encoding` at the endpoint that actually wrote the response, inspect the last bytes, and compare body completion with connection close. At an HTTPS edge, use server or proxy debug capture under an approved policy rather than assuming a network packet capture will reveal encrypted HTTP framing. The specific octet rules and completion requirement come from [RFC 9112 section 7.1](https://www.rfc-editor.org/rfc/rfc9112.html#section-7.1).

### The half-read body trap

Imagine a client that receives headers for a response with a large body, decides it no longer needs the result, and returns the connection to a pool without draining or closing it. The next borrower starts a new request, then reads bytes that actually belong to the previous body. A correct client library prevents this by draining within a limit or discarding the connection. The net effect is still important: an application that routinely abandons bodies can lose reuse and repeatedly pay cold setup. The precise behavior varies by client library, version, and body size. Instrument `new connections per completed request` and body completion rather than assuming a high-level `close()` call necessarily means socket reuse.

An early server response has the symmetric issue. If a client sent a request body and the server decides to reject it before reading all bytes, the server cannot treat remaining body bytes as a clean next request. RFC 9112 instructs it to read the entire request body or close the connection after the response. This is why a large rejected upload can coincide with a connection close even when keep-alive is enabled globally. A connection closed for framing safety is not evidence that the keep-alive setting was ignored.

## 6. `Connection: close` is a controlled diagnostic tool

When a reusable socket is the suspect, we can deliberately remove reuse for one request and compare behavior. Send `Connection: close` on an HTTP/1.1 request and inspect whether the response also advertises closure and whether a subsequent transfer dials again. This option applies to the current hop. It is not a global directive to every intermediary along the path. It is also not a command to abort the current response immediately; the connection closes *after* the current response. RFC 9112 section 9.6 defines the option.

The diagnostic question is specific: does forcing a new client-facing connection change the symptom? If a stale pooled socket was being reused after an intermediary silently timed it out, forcing close may eliminate sporadic first-byte failures at the cost of setup latency. If the symptom remains, investigate the server, upstream pool, or path. A client can still encounter a failure on a fresh connection, and the edge may still reuse its origin socket. Never turn a successful diagnostic into a permanent blanket `Connection: close` policy without pricing the extra handshakes and sockets.

![Decision tree separating client queue wait, cold setup, server work, and idle close](/imgs/blogs/http-1-1-keep-alive-and-the-six-connection-tax-7.webp)

| Observation | First discriminating measurement | Likely layer | Next safe action |
| --- | --- | --- | --- |
| First request slow, later requests fast | `num_connects` and connect/TLS timing | Cold client connection | Keep a bounded pool; compare warm and cold |
| Many concurrent requests wait before bytes leave | Browser queue phase or pool-acquire timer | Client socket budget | Count active sockets and protocol version |
| Failure after an idle gap | FIN/RST capture and idle timeout on both hops | Pool/peer timeout mismatch | Align lifetimes and verify retry semantics |
| Server trace fast, client first byte slow | Request-on-wire timestamp versus server span | Queue, proxy, or path | Capture one hop at a time |
| Body finishes but socket not reused | Framing headers and body-consumption state | HTTP message completion | Consume or close body correctly |

The table intentionally avoids numeric thresholds. An idle timeout is a policy setting, not a universal protocol constant. A “stale socket” hypothesis needs two observations: the socket was reused, and its peer or intermediary had already closed or invalidated it. A packet capture can distinguish FIN, RST, and a timeout; a client error string alone often cannot. [Health checks and graceful shutdown](/blog/software-development/networking/health-checks-draining-and-the-graceful-shutdown-nobody-implements) develops the adjacent deploy and draining case. [Timeouts, retries, and backoff](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) owns the operational policy for recovering from an allowed retry.

### Keep-alive is not a lifetime guarantee

A peer can close an idle HTTP/1.1 connection. A process restart can close it. A proxy can maintain different timers on its front and back sides. A firewall or NAT can forget state. None of these contradict persistence by default. Persistence says reuse is allowed while the connection remains valid and both sides agree on message boundaries. It does not grant an infinite lease.

The usual race looks like this: a client picks a socket from its idle pool while a peer is closing that socket, writes a request, and discovers the close only on write or read. A robust client may open a new connection and retry when method semantics make that safe. The retry decision belongs to the operation, not merely to the transport. A non-idempotent payment request should not be replayed just because the error says `connection reset`. [Idempotency keys and safe retries](/blog/software-development/api-design/idempotency-keys-safe-retries-and-exactly-once-illusions) owns the application contract that can make some retries safe.

This is also why setting the client idle lifetime a little shorter than the server or proxy idle lifetime can reduce reuse of nearly expired sockets. The exact margin must account for clock behavior, traffic, and intermediaries; it is a configuration experiment, not a protocol truth. Measure idle age at acquisition and the peer's close timing before changing every timeout in a fleet.

### A three-hop example that prevents a false fix

Consider a browser, an edge proxy, and an origin service. There are two HTTP connection boundaries: browser to edge and edge to origin. Imagine that the browser downloads two resources over the same downstream connection. The edge returns the first resource from cache but forwards the second to origin. The browser's developer tools may show one reused socket, yet the second request can still wait while the edge obtains an upstream connection. Conversely, the browser can open two sockets to the edge while the edge sends both origin requests over one already warm upstream pool. Reuse is a property of a *hop*, not a property of an end-to-end URL.

This matters when assigning responsibility. A spike in browser `time_connect` is downstream establishment. A spike after the request leaves the browser but before the origin sees it could be edge queueing, cache work, an upstream connect, or network time. An origin application span begins too late to count any of those. If an edge vendor exposes upstream connection reuse ratio and upstream connect latency, inspect them alongside downstream counters. If it does not, correlate a narrowly scoped browser-to-edge capture with an edge-to-origin capture during one request. Observe whether a new SYN occurs on each hop. The absence of a SYN on the downstream capture does not prove absence upstream.

Now suppose the origin responds with `Connection: close` to the edge while the edge keeps the browser socket alive. The browser may continue to report reuse; the edge may dial origin for every miss. Adding `Connection: keep-alive` to the browser request will not repair the upstream policy. Similarly, if the browser sends `Connection: close` to edge, that is not an instruction for the edge to close its origin pool. The hop-by-hop scope of the field explains both observations. A good incident note records protocol, connection identifiers, and close direction separately for each side of the proxy.

Connection lifetimes can produce a less obvious race. Suppose the edge believes an idle origin socket remains usable for longer than the origin does. When traffic resumes, the edge selects the stale socket and attempts a request just as the origin sends a FIN or has already closed. Depending on timing, the edge might see an immediate write error, a reset, or a response timeout. A retry may hide the event from the browser while adding tail latency. If retry is unsafe or exhausted, it becomes an error. The diagnostic comparison is not simply “set every keepalive timeout longer.” Read the configured idle limits on both sides, capture the close direction, and choose a shorter upstream pool idle limit or a robust stale-connection check. Preserve the retry's method and idempotency constraints.

This is the point where a wire-level account hands over to service architecture. [Service discovery and load balancing](/blog/software-development/microservices/service-discovery-and-load-balancing) discusses which backend the proxy picks; this post asks whether the chosen backend has an established, reusable wire path. [Observability by design](/blog/software-development/system-design/observability-metrics-logs-traces-by-design) covers how to expose the spans and metrics across all three hops. The useful division of labor is to keep the identifiers joinable: request ID, client socket ID, upstream socket ID, chosen backend, and connect outcome. Without them, a fast origin trace and slow browser report remain impossible to reconcile.

### A request that appears to hang after the headers

Framing bugs can masquerade as keep-alive bugs. Consider a server that sends a status line and headers, promises a body of a given length, but stops after fewer bytes without closing promptly. The client waits for the missing bytes because its parser has not reached the declared boundary. A trace might show a quick first byte and a long total time. Lowering the keep-alive timeout may make the hang end sooner, but it does not make the response complete. RFC 9112 calls the prematurely closed response incomplete. The correct fix is to send the promised body length, correct the declared length, or use valid chunked framing for a streaming body.

The inverse is also dangerous. A server that writes extra bytes after a declared body can leave a persistent connection with bytes that the client may interpret as the next response. A robust client may discard the connection, but the underlying mismatch remains. This is why debugging should record the actual wire framing, not only the application object's intended payload size. Look at raw headers and bytes on the specific hop. Be careful with packet captures: TLS hides HTTP framing until a legitimate endpoint or proxy decrypts it, and production payloads can contain sensitive data. The local cleartext lab avoids that exposure while demonstrating the same HTTP rule.

## 7. A worked design review for an asset-heavy page

Suppose a team proposes to speed a legacy HTTP/1.1 page by moving half its images to a second hostname. Start with the critical path, not the hostname count. Are the resources simultaneously discoverable from the initial HTML, or do scripts reveal them later? Are they small enough that one connection's response time is dominated by RTT, or large enough that bandwidth is the limiting resource? Is the second hostname already connected? Will the browser actually create six additional sockets under its current policy? Does the edge treat both hostnames as one backend pool or two? Is the page served over HTTP/1.1 to the client, or has HTTP/2 already removed the primary reason to shard?

The modeled $200\,\mathrm{ms}$ versus $260\,\mathrm{ms}$ calculation above answers only one narrow version of that review. It tells us that, with a cold second hostname costing $160\,\mathrm{ms}$ and an avoided queue wave of $100\,\mathrm{ms}$, sharding loses by $60\,\mathrm{ms}$ on the last resource's critical path. If the second hostname's connection was warm, the same model changes to one $100\,\mathrm{ms}$ wave and sharding wins by $100\,\mathrm{ms}$. Both results are derived from assumptions. Neither should be reported as measured page performance. A page waterfall under the target browser and network condition decides which model resembles reality.

The same discipline applies to a service-to-service HTTP client. If an RPC pool is capped at six connections, does it really need six? The answer depends on the number of concurrently outstanding operations, service-time distribution, and whether the client can share a socket while a response body is streaming. Use Little's Law as an explanatory capacity estimate: $L \approx \lambda W$, where $L$ is average operations in the system, $\lambda$ is completed operations per second, and $W$ is average time each occupies a slot. At $100\,\mathrm{requests/s}$ and $0.05\,\mathrm{s/request}$, the average occupied slots are (5). A pool of five has no headroom for variability; a pool of six still may queue at the tail. Little's Law does not itself choose a safe maximum or model the burst distribution. [Connection pools and tail latency](/blog/software-development/networking/connection-pools-and-where-your-tail-latency-actually-lives) handles sizing and queueing in depth.

Avoid turning this into an architecture debate about how many services or CDNs the product should have. [Load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) owns the system-level trade-off. Our wire-level question is narrower: which socket accepted each request, when was that socket ready, when did bytes move, and what framed the response so the socket could be reused?

## 8. Capture the evidence without guessing

Start with protocol negotiation. A browser may speak HTTP/2 or HTTP/3 even when the backend is HTTP/1.1. A server access log that says HTTP/1.1 is therefore not proof that the browser paid an HTTP/1.1 six-socket queue. Inspect the browser's network protocol column or a client-visible ALPN result, then inspect the edge-to-origin hop separately. This is the same boundary warning MDN makes for hop-by-hop connection management.

For a command-line reproduction against a known HTTP/1.1 endpoint, curl's verbose mode can show a new connection and later reuse *within one curl process*. Running `curl` as two separate shell processes generally creates two separate client pools, so it is a poor test of reuse. Use two URLs in one invocation, or a persistent client object in your application language. Also avoid concluding that `Connection: keep-alive` in a response proves the socket was actually reused. It only describes connection intent on that hop. The second request and socket identity are the evidence.

A small capture of the lab path can count SYNs and correlate them with requests. Filter by the exact lab host and port so you do not collect unrelated traffic:

```bash
sudo ip netns exec c tcpdump -ni c0 -s 128 -c 80 'host 10.77.0.2 and tcp port 8080'
```

The `-s 128` snap length is enough for basic TCP flags and some headers but may truncate application content. A capture can still contain sensitive request data, so use only the lab or a reviewed production capture policy. One SYN for two completed requests on one connection supports reuse on that hop. Two SYNs support two separate transport establishments. The packet count alone does not prove which application chose the close, so pair it with `Connection` headers, FIN/RST direction, and application logs.

At the server, `ss -tn` can show established sockets, and access logs can include connection identity if the server exposes it. At a proxy, separate downstream from upstream socket metrics. A high downstream reuse ratio and low upstream reuse ratio points to the proxy-to-origin hop. A high connection count with low request rate can indicate idle pool overprovisioning, slow bodies, or long-lived streams. Do not collapse those into a single “network slow” diagnosis.

The question for every dashboard is whether it counts requests, *active* connections, or *new* connections. These are different quantities. Connection establishment rate can rise while request rate stays flat if reuse deteriorates. Established connection count can rise while new connection rate stays flat if clients keep idle sockets longer. Pool queue time can rise while both counts stay constant if response occupancy grows. Recording all three exposes the mechanism much faster than a generic latency histogram.

### Add a connection identity to a service client

Browser tools are useful for pages, but many incidents begin in a service client. If the client is written in Go, `net/http/httptrace` exposes connection acquisition events for one request. The following complete program uses a single `http.Client` for two requests. Its trace prints whether each request received a reused connection and whether a new dial began. Pass a URL to a test endpoint you control, ideally the lab server in the next section. It forces HTTP/1.1 by disabling HTTP/2 attempts for this plain HTTP URL; it does not disable pooling. Close and fully read each response body so the transport may return the connection to its idle pool.

```go
package main

import (
    "context"
    "fmt"
    "io"
    "net/http"
    "net/http/httptrace"
    "os"
    "time"
)

func main() {
    if len(os.Args) != 2 {
        fmt.Fprintln(os.Stderr, "usage: go run main.go http://10.77.0.2:8080/")
        os.Exit(2)
    }
    tr := &http.Transport{
        ForceAttemptHTTP2: false,
        MaxIdleConnsPerHost: 2,
        IdleConnTimeout: 30 * time.Second,
    }
    defer tr.CloseIdleConnections()
    client := &http.Client{Transport: tr, Timeout: 3 * time.Second}
    for i := 1; i <= 2; i++ {
        req, err := http.NewRequest(http.MethodGet, os.Args[1], nil)
        if err != nil { panic(err) }
        trace := &httptrace.ClientTrace{
            GetConn: func(host string) {
                fmt.Printf("request=%d GetConn host=%s\n", i, host)
            },
            ConnectStart: func(network, addr string) {
                fmt.Printf("request=%d ConnectStart %s %s\n", i, network, addr)
            },
            GotConn: func(info httptrace.GotConnInfo) {
                fmt.Printf("request=%d GotConn reused=%t was_idle=%t\n",
                    i, info.Reused, info.WasIdle)
            },
        }
        req = req.WithContext(httptrace.WithClientTrace(context.Background(), trace))
        resp, err := client.Do(req)
        if err != nil { panic(err) }
        _, readErr := io.Copy(io.Discard, resp.Body)
        closeErr := resp.Body.Close()
        if readErr != nil { panic(readErr) }
        if closeErr != nil { panic(closeErr) }
        fmt.Printf("request=%d status=%s\n", i, resp.Status)
    }
}
```

A normal local result has `ConnectStart` on the first request, `reused=false` for its `GotConn`, and `reused=true` on the second. If the server sends `Connection: close` or the client fails to consume a body, expect another dial. This is an expected qualitative result, not a claim that the program was run against a production endpoint. The program prints socket-acquisition events, but it does not directly print how long a request waited for a free pool slot. For that, timestamp `GetConn` and `GotConn` with a monotonic clock and subtract them. Also record dial start and completion so pool wait is not mislabeled as TCP setup. The [Go httptrace documentation](https://pkg.go.dev/net/http/httptrace) defines those hooks and their scope.

Be cautious with the meaning of `reused=true`. It means the transport used a connection previously used for a completed request, according to that client's transport. It does not prove the edge reused its upstream origin connection. A local TCP capture can verify the client-facing socket; edge metrics or a second capture are needed for the upstream hop. This is another reason to attach a hop label to every connection metric.

## 9. Run it yourself

### Question

Does one HTTP/1.1 client process reuse a completed connection for a second request, and does `Connection: close` force the next request to dial a new connection? This lab tests connection reuse on the client-to-server hop, not browser socket limits or TLS setup. It keeps protocol and payload fixed while changing only the connection option.

### Preconditions

Use a disposable Linux environment with the `c` and `s` namespaces from [the series setup](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url), connected by `c0` and `s0`, with client `10.77.0.1` and server `10.77.0.2`. The repository may not include a prebuilt `netlab` tree, so this experiment starts a temporary Python standard-library HTTP/1.1 server directly in namespace `s`. You need Linux, Python 3, curl with `num_connects` write-out support, iproute2, and root or `CAP_NET_ADMIN` for `ip netns exec`. Check versions with `python3 --version`, `curl --version`, and `ip -Version`. The commands below assume the two namespaces already exist; they do not create or delete them. Run them only in the disposable lab. No production interface or firewall rule is changed.

First verify the path and the listener port before starting this experiment:

```bash
set -euo pipefail
ip netns list | grep -E '^(c|s)( |$)'
ip netns exec c ip -brief address show dev c0
ip netns exec s ip -brief address show dev s0
ip netns exec c ip route get 10.77.0.2
ip netns exec s ss -lnt '( sport = :8080 )'
ip netns exec c ping -c 2 -W 1 10.77.0.2
```

The `ss` line should show no listener on port 8080 before this self-contained experiment. If a previous lab owns that port, stop it using that lab's own cleanup; do not kill an unspecified process. The route should choose `c0`; ping should receive replies. Ping proves only basic reachability, not HTTP correctness or connection reuse.

### Baseline

Run this in one shell. The temporary server deliberately responds with a self-defined `Content-Length` and stays on HTTP/1.1. It logs each accepted TCP connection and each request. Keep the server's PID file so reset is scoped.

```bash
set -euo pipefail
LAB_DIR=$(mktemp -d /tmp/http11-reuse.XXXXXX)
cat > "$LAB_DIR/server.py" <<'PY'
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def do_GET(self):
        body = b"ok\n"
        self.send_response(200)
        self.send_header("Content-Type", "text/plain")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)
        self.wfile.flush()
        print(f"request path={self.path} peer={self.client_address}", flush=True)

    def log_message(self, format, *args):
        pass

class Server(ThreadingHTTPServer):
    def get_request(self):
        sock, address = super().get_request()
        print(f"accepted peer={address}", flush=True)
        return sock, address

Server(("10.77.0.2", 8080), Handler).serve_forever()
PY
printf '%s\n' "$LAB_DIR" > /tmp/http11-reuse-active-dir
ip netns exec s python3 "$LAB_DIR/server.py" > "$LAB_DIR/server.log" 2>&1 &
echo $! > "$LAB_DIR/server.pid"
sleep 1
ip netns exec s ss -lnt '( sport = :8080 )'
ip netns exec c curl --http1.1 --noproxy '*' -sS -o /dev/null \
  -w 'first: connects=%{num_connects} connect=%{time_connect} total=%{time_total}\n' \
  http://10.77.0.2:8080/one \
  --next --http1.1 --noproxy '*' -sS -o /dev/null \
  -w 'second: connects=%{num_connects} connect=%{time_connect} total=%{time_total}\n' \
  http://10.77.0.2:8080/two
cat "$LAB_DIR/server.log"
```

Read `connects` on each transfer and count `accepted` lines. Expected state on a curl build that reuses a completed HTTP/1.1 socket across the `--next` transfer: first `connects=1`, second `connects=0`, and one accepted TCP peer for two requests. `time_total` is not given a tight numeric range because this localhost namespace path has scheduler and virtualization noise; expect two successful transfers and normally subsecond totals on an unloaded local VM. If your curl version or option handling creates a second connection, inspect `curl -v` before treating the output as a protocol violation. `--next` resets local options, so the command repeats `--http1.1` and `--noproxy` explicitly.

### Apply one change

Repeat the same two-request shape with only the request's connection option changed. The `Connection: close` header on the first transfer asks the server to close after that response. The second transfer must therefore use a new TCP connection. We leave body, path pattern, and server unchanged.

```bash
set -euo pipefail
LAB_DIR=$(cat /tmp/http11-reuse-active-dir)
: > "$LAB_DIR/server.log"
ip netns exec c curl --http1.1 --noproxy '*' -sS -o /dev/null \
  -H 'Connection: close' \
  -w 'first: connects=%{num_connects} connect=%{time_connect} total=%{time_total}\n' \
  http://10.77.0.2:8080/one \
  --next --http1.1 --noproxy '*' -sS -o /dev/null \
  -w 'second: connects=%{num_connects} connect=%{time_connect} total=%{time_total}\n' \
  http://10.77.0.2:8080/two
cat "$LAB_DIR/server.log"
```

### Compare

Read `num_connects`, the number of `accepted` lines, and the server's two request paths. Expected state: the treatment has two accepted connections, with `connects=1` for each transfer. The second `time_connect` should be nonzero on most curl builds, although its precise magnitude is not a stable performance benchmark. In the baseline the second transfer could reuse a connection because the first response had a known three-byte body and was fully consumed. In the treatment the first transfer explicitly ended the connection's useful life. That is the claim under test.

For packet-level confirmation, repeat either command while a second terminal runs `ip netns exec c tcpdump -ni c0 -s 128 'host 10.77.0.2 and tcp port 8080'`. Compare SYN packets with the server's `accepted` lines. Packet capture may include request data and requires appropriate privileges; use only the disposable lab. The baseline should show one client SYN across the pair, while the treatment should show two, unless an unrelated local client is using the port. The exact packet count and timings vary with kernel and environment; the SYN difference is the reproducible property.

### Reset

```bash
set -euo pipefail
LAB_DIR=$(cat /tmp/http11-reuse-active-dir)
kill "$(cat "$LAB_DIR/server.pid")"
rm -f /tmp/http11-reuse-active-dir
rm -rf -- "$LAB_DIR"
ip netns exec s ss -lnt '( sport = :8080 )'
```

Reset removes only the temporary server and its directory. It does not delete namespaces, interfaces, routes, or qdiscs owned by the series setup. On a production host, use read-only client timing, a narrowly scoped `ss -tn` query, connection counters, and application pool-acquire metrics. Do not send `Connection: close` broadly in production merely to reproduce this lab's effect.

### Interpret the result before changing a fleet setting

This lab is deliberately narrower than a browser waterfall. It proves that a self-defined response length permits a subsequent request to reuse one HTTP/1.1 socket in this client and server pair. It also proves that an explicit close ends that opportunity. It does not establish a browser's per-origin socket limit, the benefits of TLS resumption, or a safe idle timeout for your production proxy. Those require different experiments. Keeping the claim narrow is what makes the result useful.

If the baseline unexpectedly shows two connections, first check whether the server emitted an HTTP/1.1 response with the expected length, whether curl actually ran both transfers in one process, and whether a proxy environment variable redirected traffic. The command uses `--noproxy '*'` to rule out a proxy for the lab IP. Next inspect verbose curl output and the server's accepted peers. If curl reports reuse but the server logs two accepts, you may be correlating separate runs or reading a stale log. If the server logs one accept but the second transfer fails, inspect body completion and whether the server closed after the first response. A mismatch is evidence to investigate, not a reason to change the expected result until every layer is identified.

The treatment's timing may appear almost indistinguishable from baseline on a local veth pair. That is expected: the packet path can have a very small RTT, so the extra handshake need not produce a striking wall-clock difference. This experiment identifies connection behavior through `num_connects`, accepted socket count, and SYN packets. To study latency impact, add a *measured* RTT impairment to the disposable `netlab` path, rerun both cases, and report the actual `ping` RTT and median across repeated trials. Do not silently assume a requested one-way `tc netem` delay equals a particular round-trip time; where the qdisc is installed determines which traffic direction it delays. Remove only the qdisc that your experiment added. The observed connection count should stay the same while timing differences become easier to see.

## 10. The decisions that survive a protocol upgrade

When a browser speaks HTTP/2, the familiar six HTTP/1.1 sockets may disappear from the client-facing hop, but connection management does not disappear. Streams share flow-control windows and a TCP connection; an intermediary may still speak HTTP/1.1 upstream. When a client speaks HTTP/3, QUIC changes transport stream behavior, but it still maintains connection state, idle timers, and retries. The transferable habit is to name the hop and protocol for every claimed bottleneck.

Use a persistent HTTP/1.1 connection when multiple requests will reach the same hop and message boundaries are unambiguous. Keep the pool bounded and measure queue wait. Force `Connection: close` for a narrow diagnostic comparison or for a known compatibility constraint, then remove it when the experiment ends. Avoid domain sharding as a reflex; calculate the setup cost and verify the actual client protocol first. Treat pipelining as a historical and standards mechanism, not an easy performance flag for modern browsers.

The next time a waterfall shows “stalled,” place the stall before or after the first byte leaves the client. The next time a server trace is fast while the client is slow, ask whether the request was waiting for a socket. The next time a connection resets after an idle period, inspect which hop closed it and whether the operation can be retried. Those three questions locate most of the supposed mystery in ordinary, measurable state.

## Further reading

- [RFC 9112: HTTP/1.1, June 2022](https://www.rfc-editor.org/rfc/rfc9112.html), especially message body length, chunked transfer coding, persistence, pipelining, and `Connection: close`.
- [MDN: Connection management in HTTP/1.x](https://developer.mozilla.org/en-US/docs/Web/HTTP/Guides/Connection_management_in_HTTP_1.x), for the browser-oriented explanation of persistence, pipelining, and sharding.
- [Firefox 54 developer release notes](https://developer.mozilla.org/en-US/docs/Mozilla/Firefox/Releases/54) and [Mozilla bug 1340655](https://bugzilla.mozilla.org/show_bug.cgi?id=1340655), for the dated pipelining removal case.
- [RFC 8446: TLS 1.3, August 2018](https://www.rfc-editor.org/rfc/rfc8446.html), for the handshake structure underlying the cold HTTPS model.
- [curl write-out documentation](https://curl.se/docs/manpage.html#-w), for the timing and connection fields used in the lab.
