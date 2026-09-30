---
title: "WebSockets, SSE, and the Long-Lived Connection You Actually Have to Operate"
date: "2026-09-30"
publishDate: "2026-09-30"
description: "Trace an upgraded connection through proxies, bound its queues, and design reconnect and replay paths that survive a restart."
tags:
  ["networking", "distributed-systems", "websocket", "server-sent-events", "realtime", "backpressure", "reconnection", "proxies", "fan-out", "event-streams"]
category: "software-development"
subcategory: "Networking"
author: "Hiep Tran"
featured: true
readTime: 41
image: "/imgs/blogs/websockets-sse-and-long-lived-connections-1.webp"
---

The chat service works after deploy, then a suspicious fraction of clients disconnect at nearly the same interval. A dashboard says the WebSocket upgrade succeeded. The handler is healthy. Yet presence blinks, browser consoles report close code `1006`, and a second wave of new connections arrives just when the new pods are warming. The successful `101 Switching Protocols` response is only the first event in the life of this connection. The failure can occur minutes later in a proxy with a shorter idle timer, in an unbounded per-client queue, or during reconnection when the server asks every browser to rebuild state at once.

![Latency ladder contrasting opening costs with warm tunnel event delivery and teardown](/imgs/blogs/websockets-sse-and-long-lived-connections-1.webp)

The diagram above is the mental model: DNS, TCP, and TLS are opening costs; a warm tunnel keeps delivering frames until an endpoint or intermediary closes it. DNS supplies an address, but data packets do not pass through the resolver. A TCP and usually TLS connection reaches an edge or load balancer, then an HTTP-aware proxy sees the upgrade request. Once the proxy accepts it, both peers expect to exchange frames for as long as the product needs, while each intervening hop still enforces its own connection and idle policy. A successful opening does not prevent a proxy idle timer from ending the warm tunnel. To place these stages in the wider request path, start with [what happens when you curl a URL](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url). The [senior engineer's network mental model](/blog/software-development/networking/the-senior-engineers-network-mental-model) brings the layers back together.

This post follows five questions. What did the HTTP handshake promise? What do WebSocket frames buy after that handshake? Which hop closes a quiet connection? What happens when one subscriber is slower than the publisher? How does the service restore a useful stream after a process dies? We will also compare Server-Sent Events (SSE), which solves a narrower but often sufficient problem. The comparison is operational, not a contest for a universal winner.

## 1. A successful upgrade only changes the rules for one connection

**Rule of thumb: debug the opening handshake and the long-lived tunnel as separate phases.** A WebSocket client first makes an HTTP/1.1 request with an `Upgrade: websocket` header, a `Connection: Upgrade` token, a fresh `Sec-WebSocket-Key`, and version `13`. A server that accepts it responds with status `101`, `Upgrade: websocket`, `Connection: Upgrade`, and a `Sec-WebSocket-Accept` value calculated from the client's key and the protocol's fixed GUID. The key and accept value establish that both sides understood the upgrade; they are not authentication or encryption. [RFC 6455, published December 2011](https://www.rfc-editor.org/rfc/rfc6455#section-4) specifies the exchange.

![HTTP upgrade, data frames, ping and pong, and the closing handshake on one connection](/imgs/blogs/websockets-sse-and-long-lived-connections-2.webp)

The sequence matters because a `101` tells us only that an HTTP hop agreed to switch protocols. It does not prove that a downstream proxy will keep the connection open, that an authenticated subscription was installed, that later frames can reach the app, or that a disconnected client can resume. In a capture, separate the TCP and TLS handshake, the HTTP opening handshake, and subsequent WebSocket frames. That distinction also prevents a common timing mistake: `curl -w` can measure the opening HTTP transaction, but it does not report the lifetime or delivery latency of a WebSocket subscription.

A minimal request, with an illustrative key, has this shape:

```http
GET /updates HTTP/1.1
Host: example.test
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
Origin: https://example.test
```

The well-known sample key comes from [RFC 6455 section 1.3](https://www.rfc-editor.org/rfc/rfc6455#section-1.3), so it is useful for explaining bytes but must never be reused as a client's fresh random key. A normal browser generates its own handshake. Behind nginx, `Upgrade` and `Connection` are hop-by-hop HTTP headers. A reverse proxy does not blindly forward them as ordinary end-to-end headers; its upstream configuration must explicitly pass the upgrade intent. The [nginx WebSocket proxy documentation](https://nginx.org/en/docs/http/websocket.html) shows both `proxy_set_header Upgrade $http_upgrade` and a mapped `Connection` value. Its current page notes a historical version detail: explicit `proxy_http_version 1.1` was needed before nginx 1.29.7. Check the installed version before copying a version-specific stanza.

Do not overlearn the HTTP/1.1 part. [RFC 8441, published September 2018](https://www.rfc-editor.org/rfc/rfc8441) defines an extended `CONNECT` path for WebSockets over HTTP/2. The route actually negotiated by a browser, intermediary, and origin determines the wire exchange. Classic Upgrade is still the clearest debugging starting point, but the protocol is not conceptually tied to an HTTP/1.1 TCP connection forever. The companion [HTTP/2 multiplexing post](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer) explains why an HTTP/2 stream and its transport connection have different lifetimes.

The moment after the upgrade is easy to misunderstand. WebSocket is a framed protocol over the established transport. It does not turn the connection into raw application bytes. Frames have a FIN bit, opcode, payload length encoding, and optional masking key. An application message can occupy one frame or several fragments. Intermediaries can change TCP segmentation; a TCP packet is not a WebSocket message boundary. A `send()` call is not evidence that the peer application consumed a message. It usually means the local stack accepted data into one of several buffers.

Clients must mask frames they send to servers, with a new unpredictable 32-bit key per frame; servers must leave their frames unmasked. That asymmetry is a protocol defense against old proxy behaviors, not a confidentiality mechanism. TLS is what protects contents in transit when `wss://` is used. A server must reject an unmasked client frame; a client must reject a masked server frame. These are requirements of [RFC 6455 sections 5.1 and 5.3](https://www.rfc-editor.org/rfc/rfc6455#section-5.1). If a custom client succeeds in a local direct connection but fails through a strict gateway, inspect compliance before assuming the gateway is broken.


### Count bytes at the frame boundary, not at the message boundary

For an unfragmented payload up to 125 bytes, the base WebSocket frame header occupies 2 bytes. A browser-to-server frame also carries a 4-byte masking key, so a 100-byte client message occupies at least 106 WebSocket bytes before TLS and TCP/IP overhead. The same unmasked 100-byte server message occupies at least 102 WebSocket bytes. Those totals are derived from the [RFC 6455 base framing format](https://www.rfc-editor.org/rfc/rfc6455#section-5.2): 2+4+100 and 2+100. They are not packet sizes. TLS records, TCP segments, IP headers, and retransmissions add separate costs. For payload lengths beyond 125 bytes the framing uses extended-length fields, so the 2-byte fixed header is no longer the whole length field.

This arithmetic changes protocol decisions at small message sizes. If an app emits hundreds of tiny state changes, batching can save frame and record overhead, but batching also adds waiting time before the first state change is visible. A 20-millisecond batching window, for example, can add anywhere from nearly zero to about 20 milliseconds of application delay depending on arrival time, before network delay. That is a derived bound under a periodic flush model, not a browser guarantee. Record both bytes saved and event age before deciding that fewer frames are an improvement. If a message contains a latency-sensitive control event, keep its batch policy separate from bulk updates.

Fragmentation and compression deserve the same scope discipline. A WebSocket message can be fragmented into multiple frames, with control frames inserted between fragments. An application should cap the reassembled message size, not only individual frame size; otherwise an attacker can send many legal fragments that exhaust memory. Compression extensions can reduce bandwidth for repetitive payloads but consume CPU and may retain per-connection state. The negotiated extension and its parameters, not a generic “WebSocket supports compression” checkbox, determine behavior. Start with size limits and backpressure before using compression to mask an oversized event design. RFC 6455 describes extension negotiation and explicitly requires implementations with message-size limits to defend those limits.

Control frames are a separate small vocabulary. Opcode `0x8` means Close, `0x9` Ping, and `0xA` Pong. RFC 6455 says a control frame is at most 125 payload bytes and cannot be fragmented; a receiver normally responds to a Ping with a Pong unless it has already received Close. Control frames can be interleaved inside a fragmented data message. That is why a large message should not prevent liveness checks from being represented on the wire, though an overloaded event loop can still delay the actual reply. A proper Close exchange gives both peers a protocol outcome before the underlying TCP connection closes. An abrupt disappearance has no WebSocket Close frame; browsers commonly surface `1006` as a local abnormal-close observation. Code `1006` is reserved and cannot be sent in a Close frame. [RFC 6455 section 7.4](https://www.rfc-editor.org/rfc/rfc6455#section-7.4) defines that distinction.

The opening path also raises a security boundary. A browser sends `Origin`, but WebSocket is not made safe by ordinary cross-origin HTTP assumptions. A server should validate allowed origins for browser clients, authenticate the session at handshake or in the application protocol, authorize each subscription, and bound message size. A non-browser client can forge an Origin header, so Origin is a browser-origin defense, not identity proof. The RFC's [security considerations](https://www.rfc-editor.org/rfc/rfc6455#section-10) cover origin and implementation-specific size limits. When a proxy terminates TLS, be explicit about whether the next hop is encrypted; the [TLS termination and re-encryption post](/blog/software-development/networking/terminating-and-re-encrypting-tls-edge-mesh-and-end-to-end) owns that separate trust boundary.

## 2. The quiet connection is not necessarily an alive connection

The canonical latency ladder is `DNS | TCP | TLS | request | server think | first byte | transfer`. For a warm WebSocket subscription, the first four costs were paid earlier. The user's visible delay now lives in event creation, fan-out queues, proxy forwarding, transfer, and browser processing. A beautiful `101` timing cannot tell us whether events arrive promptly. The operational clock to draw is the time since the last byte each hop read from its upstream peer.

The [nginx WebSocket proxy guide](https://nginx.org/en/docs/http/websocket.html) says its default proxied connection closes when the upstream server sends no data for 60 seconds; `proxy_read_timeout` can change that interval, and a server Ping can reset it. This is a *read* timeout for that proxy leg. It does not mean the application is globally idle, nor does it establish the timeout of a cloud load balancer in front of nginx. A client that sends its own message may keep some hop states active while nginx still sees no upstream bytes. Diagnose the direction and owner of each timer.

<figure class="blog-anim">
<svg viewBox="0 0 820 300" role="img" aria-label="Two upgraded WebSocket tunnels: the silent one reaches the proxy idle timeout while Ping traffic repeatedly resets the other timer" style="width:100%;height:auto;max-width:820px">
<style>
.ws3-bg{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}.ws3-text{font:600 16px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937)}.ws3-small{font:14px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280)}.ws3-line{stroke:var(--border,#d1d5db);stroke-width:4}.ws3-ping{fill:var(--accent,#6366f1)}.ws3-timer{fill:var(--accent,#6366f1)}.ws3-stop{fill:#ef4444}
@keyframes ws3-idle{0%{transform:scaleX(0)}70%,100%{transform:scaleX(1)}}
@keyframes ws3-reset{0%,28%{transform:scaleX(0)}24%{transform:scaleX(.75)}48%{transform:scaleX(.65)}50%{transform:scaleX(0)}74%{transform:scaleX(.65)}76%{transform:scaleX(0)}100%{transform:scaleX(.7)}}
@keyframes ws3-pulse{0%,20%,28%,45%,53%,70%,78%,100%{opacity:0}23%,48%,73%{opacity:1}}
@keyframes ws3-close{0%,68%{opacity:0}72%,100%{opacity:1}}
.ws3-idle{transform-origin:0 0;animation:ws3-idle 12s linear infinite}.ws3-reset{transform-origin:0 0;animation:ws3-reset 12s linear infinite}.ws3-pulse{animation:ws3-pulse 12s ease-in-out infinite}.ws3-close{animation:ws3-close 12s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ws3-idle,.ws3-reset,.ws3-pulse,.ws3-close{animation:none}.ws3-idle{transform:scaleX(1)}.ws3-reset{transform:scaleX(.4)}.ws3-close{opacity:1}.ws3-pulse{opacity:1}}
</style>
<text class="ws3-text" x="25" y="31">The proxy watches upstream silence, not application intent</text>
<rect class="ws3-bg" x="20" y="55" width="780" height="100" rx="10"/>
<text class="ws3-text" x="40" y="83">Silent tunnel</text><text class="ws3-small" x="40" y="108">No bytes from upstream</text>
<line class="ws3-line" x1="260" y1="105" x2="555" y2="105"/>
<rect class="ws3-timer ws3-idle" x="260" y="96" width="295" height="18" rx="6"/>
<text class="ws3-small" x="580" y="100">idle threshold</text><text class="ws3-text ws3-stop ws3-close" x="580" y="127">proxy closes</text>
<rect class="ws3-bg" x="20" y="175" width="780" height="100" rx="10"/>
<text class="ws3-text" x="40" y="203">Ping on tunnel</text><text class="ws3-small" x="40" y="228">Traffic resets read timer</text>
<line class="ws3-line" x1="260" y1="225" x2="555" y2="225"/>
<rect class="ws3-timer ws3-reset" x="260" y="216" width="295" height="18" rx="6"/>
<circle class="ws3-ping ws3-pulse" cx="450" cy="203" r="9"/>
<text class="ws3-small" x="580" y="220">below threshold</text><text class="ws3-text" x="580" y="247">tunnel survives</text>
</svg>
<figcaption>An upstream Ping can reset a proxy read-idle timer; a silent upgraded tunnel may close when that timer expires. The threshold is conceptual and depends on proxy configuration.</figcaption>
</figure>

A heartbeat is traffic chosen to make liveness and idle policy observable. A server-side WebSocket Ping is attractive when the upstream-to-proxy read timer is the problem, because it traverses the exact leg the proxy watches. A browser JavaScript `WebSocket` API does not expose Ping-frame creation, so a browser client may instead send an application heartbeat message if the application protocol requires client-originated liveness. Those are different mechanisms. A transport-level Pong response says the peer's protocol stack handled Ping; it is not proof that a downstream database or business operation is healthy.

Suppose a proxy has a 60-second upstream read timeout, and the server sends a Ping every 25 seconds. In the absence of unusual scheduling stalls, each Ping arrives well before the 60-second gap. The nominal margin is $60-25=35$ seconds. This is a derived example, not a universal safe interval: event-loop pause, proxy buffering, network outage, and separate downstream idle policies can consume the margin. If a load balancer in front expires at 20 seconds of no traffic in either direction, the 25-second design still fails. Inventory all hops and set a heartbeat below the shortest relevant idle threshold with room for jitter. Increasing every timeout indefinitely can leave dead connections consuming sockets and memory for longer.

A heartbeat is also an assertion you can test. Record last application event time, last Ping sent, last Pong received, close code if a Close frame arrived, and the transport end reason. If disconnections cluster near a configured idle period after the last server-to-proxy byte, suspect proxy policy. If they coincide with CPU pauses while Ping scheduling is late, inspect the event loop and queue. If they happen near a fixed absolute connection age regardless of activity, an absolute lifetime policy may be involved. A capture at the proxy boundary can separate these cases, but capture only the traffic needed and protect tokens and payload data; captures can contain sensitive information.

SSE has the same long-lived-hop problem in HTTP clothing. Its server can send a comment line such as `: heartbeat` followed by a blank line without dispatching an application event. The [WHATWG SSE specification](https://html.spec.whatwg.org/multipage/server-sent-events.html) and [MDN's usage guide](https://developer.mozilla.org/en-US/docs/Web/API/Server-sent_events/Using_server-sent_events) describe this format and note that comments can prevent timeouts. At an HTTP reverse proxy, buffering can make the origin write promptly while the browser receives nothing. nginx's [proxy buffering documentation](https://nginx.org/en/docs/http/ngx_http_proxy_module.html#proxy_buffering) says disabling buffering forwards responses synchronously as received, and `X-Accel-Buffering: no` can control it unless configured to ignore that header. Do not casually disable buffering for an entire site because one SSE route needs streaming behavior.

## 3. SSE is a one-way stream with a useful browser contract

**Choose the shape of communication before choosing the fashionable protocol.** A WebSocket carries application messages in both directions over one upgraded connection. SSE is an HTTP response with MIME type `text/event-stream`, kept open while the server writes UTF-8 event records. The browser's `EventSource` receives them. Client-to-server actions remain ordinary HTTP requests. Short polling repeatedly makes a request and receives a complete response, whether or not new data exists. All three can deliver a notification; they differ in who can initiate data, how reconnect happens, and where buffering is visible.

![Comparison matrix for WebSocket, SSE, and short polling at the browser and proxy boundaries](/imgs/blogs/websockets-sse-and-long-lived-connections-4.webp)

The matrix is a design aid, not a ranking. SSE is often enough for job progress, feeds, dashboards, and server-originated notifications where a client sends infrequent commands through `POST`. The HTTP request shape fits many existing authentication and observability systems, although a stream still occupies resources for its lifetime. WebSocket is justified when low-latency messages flow frequently in both directions, such as interactive editing or an application-level control channel. Polling is perfectly reasonable when updates are infrequent, latency tolerance is broad, or intermediary behavior makes persistent streaming expensive to operate. The [API performance post](/blog/software-development/api-design/api-performance-payload-size-compression-and-tail-latency) handles payload and request-level trade-offs; this section owns stream lifetime.

An SSE record can contain `event:`, `data:`, `id:`, and `retry:` fields, separated from the next record by a blank line. Multiple `data:` lines become one event payload with intervening newlines. An `id:` value updates the browser's last event ID; after a reconnect, the browser sends `Last-Event-ID` when that value is nonempty. The WHATWG [processing model](https://html.spec.whatwg.org/multipage/server-sent-events.html#the-eventsource-interface) defines those behaviors. The server is responsible for storing events, interpreting the cursor, and deciding whether it can replay the gap. `Last-Event-ID` is a request for continuity, not a magical durable queue in the browser. If a server has already discarded the requested history, it needs an explicit application-level resync path.

A small response illustrates the syntax:

```http
HTTP/1.1 200 OK
Content-Type: text/event-stream
Cache-Control: no-cache

id: 8472
event: order-status
data: {"order":"a7","state":"packed"}

: heartbeat

```

The values `8472` and `a7` are illustrative application data, not a reported event. The blank line after the `data:` field completes one event; the comment line does not create a user event. If the browser reconnects after processing ID `8472`, it can send `Last-Event-ID: 8472`; an origin with retained ordered events can resume after that cursor. Use an opaque cursor if your storage order is not a simple integer. Do not assume a timestamp alone is a safe total ordering when multiple producers write concurrently.

The browser `EventSource` API also makes choices for you. It reconnects according to its processing model, using a reconnection delay that a server can influence with a `retry:` field; the user agent may add more delay after failures. It does not provide a bidirectional application message channel. Native `EventSource` construction is URL based and has different control over request headers than a general `fetch` call; design authentication accordingly. Cookies may be appropriate where same-site policies and CSRF defenses are already correct. Putting a long-lived bearer token in a query string is often operationally awkward because URLs can be logged; avoid turning a protocol selection into an accidental credential-handling change.

SSE over HTTP/1.1 can meet browser per-origin connection limits when several tabs each open a stream. With HTTP/2, streams can share a transport connection, subject to the negotiated concurrent-stream limit and intermediary behavior. This is a capacity consideration rather than a universal fixed number; browser and server settings matter. The mechanism is the same one covered in [HTTP/1.1 keep-alive and connection limits](/blog/software-development/networking/http-1-1-keep-alive-and-the-six-connection-tax) and [HTTP/2 multiplexing](/blog/software-development/networking/http-2-multiplexing-flow-control-and-the-hol-that-moved-down-a-layer). Measure actual connections per origin and active HTTP/2 streams for your clients instead of repeating an old browser limit as a timeless production fact.

Neither SSE nor WebSocket gives the application exactly-once delivery for free. A reconnect can race with an event that was committed but not acknowledged to the client, or acknowledged locally but not durably stored. The useful target is usually an ordered, replayable stream with idempotent event handling. Give each event a stable ID, persist before publication if it must survive restart, allow a client to request everything after its last applied ID, and make applying the same ID twice safe. The browser cursor is only one part of that contract. When the available log starts later than the requested ID, return a clear reset instruction and a snapshot path.

## 4. Backpressure starts where the fastest writer meets the slowest reader

An upgraded connection has several queues: the producer's work queue, broker delivery buffers, the server's per-client output queue, user-space or TLS buffers, kernel send buffers, proxy buffers, network flight, browser receive buffers, and the browser application's event handler. “The socket is writable” says very little about the age of data at the last consumer. If one phone goes offline on a train while a room emits frequent updates, the server may continue accepting messages into a queue that cannot drain. A large queue makes disconnection look like recovery until memory runs out or the client finally receives stale updates.

![Derived slow-subscriber queue growth and the point where a bounded policy must act](/imgs/blogs/websockets-sse-and-long-lived-connections-5.webp)

Here is an explanatory fluid model, not a WebSocket protocol equation. Let $\lambda$ be bytes arriving for one subscriber per second, $\mu$ be bytes the subscriber path drains per second, $B_0$ be queued bytes at the start, and $B_{\max}$ be the configured queue cap. While $\lambda>\mu$, approximate queued bytes after $t$ seconds as

$$
B(t) \approx \min\!\left(B_{\max}, B_0 + (\lambda-\mu)t\right).
$$

The approximation ignores burst size and scheduling, so a real queue needs headroom and measurements. Suppose an application emits 40 events per second, each 2 KiB after encoding, for $\lambda=80$ KiB/s. A slow subscriber drains 20 KiB/s. Net growth is 60 KiB/s. A 6 MiB per-client cap fills in roughly $6\times1024/60=102.4$ seconds from empty. At 10,000 similarly slow subscribers, the nominal cap represents 60,000 MiB, about 58.6 GiB, before objects, TLS state, kernel buffers, or brokers. Every figure in this paragraph is derived from the stated assumptions; it is not a production benchmark. It shows why “just increase the queue” is not a free reliability fix.

If the producer is bursty, average arrival rate hides short overload. Apply the same accounting over a burst window, and monitor oldest queued event age as well as byte count. Queue age answers the product question: is this update still useful when it arrives? A trading ticker may be able to coalesce intermediate prices to the latest state. A financial ledger cannot silently discard entries; it may need to pause delivery, spill durably, or disconnect the client and require replay. A chat room can preserve a durable sequence in storage and drop the live connection once it falls behind, then let the client catch up. The correct policy follows event semantics, not the WebSocket API.

Browser `WebSocket.send()` can queue bytes locally. MDN's [`bufferedAmount` documentation](https://developer.mozilla.org/en-US/docs/Web/API/WebSocket/bufferedAmount) defines it as queued bytes not yet transmitted to the network. That counter is useful as a guardrail for the browser's sending side, but it does not report the server's queue or confirm peer consumption. MDN's [WebSocket API overview](https://developer.mozilla.org/en-US/docs/Web/API/WebSockets_API) notes that the classic browser `WebSocket` interface has no built-in backpressure, so a fast arrival stream can overwhelm client memory or CPU. A receive handler that cannot keep pace needs application-level sampling, coalescing, worker processing, or a protocol that supports explicit credit. On the server, a write that returns successfully may only have handed data to a local buffer; apply a bounded queue and observe actual drain.

A practical server policy declares three thresholds. A soft threshold starts coalescing or suppressing replaceable updates. A hard byte and age threshold stops enqueueing and closes or pauses that subscriber. A recovery threshold allows it to rejoin only after it has replayed from a durable cursor or received a fresh snapshot. Make thresholds per client and global. If each client cap is 1 MiB and there are 50,000 simultaneous clients, the upper bound from that one category is about 48.8 GiB, derived as $50{,}000/1024$ GiB, before other memory. The global cap and eviction policy prevent many individually “safe” clients from exhausting a node together.

The same reasoning applies to SSE. Its HTTP stream writes can accumulate in user space, proxy buffering, and the browser. The protocol gives a handy event cursor, not automatic slow-consumer control. If a reverse proxy buffers SSE, the origin may see fast writes while users see delayed batches. Observe event ID and enqueue time at origin, last byte time at proxy, and receive time at browser to locate the delay. If the client is slow after the proxy, disabling origin-side buffering does not make its CPU faster. The higher-level queue policy is explained in [rate limiting and backpressure](/blog/software-development/system-design/rate-limiting-and-backpressure); the network-specific point is where the bytes wait and which hop's counter can see them.

When writing load tests, do not report only median event latency from healthy subscribers. Partition clients into fast, slow, and disconnected cohorts; include active connections, queued bytes by percentile, oldest queue age, dropped/coalesced event count, and reconnect rate. A benchmark with one local subscriber cannot validate a policy for a thousand geographically dispersed browsers. State payload size, message rate, concurrency, RTT, proxy version, and duration. If the test client discards received data without parsing or rendering it, it does not reproduce browser backpressure. The measurement setup is part of the claim.

## 5. Reconnect is a load-generating operation

A long-lived connection will end. Devices sleep, Wi-Fi networks change, proxies rotate, processes deploy, and entire zones fail. A robust client therefore has a state machine, not an `onclose` handler that immediately calls `new WebSocket()` forever. Distinguish connecting, authenticated, subscribed, receiving, resuming, and needing a full snapshot. A new transport connection is only the first step. If the old node carried subscription and delivery position solely in memory, a new node cannot infer what the browser missed.

Consider 100,000 connected clients that all lose their sockets after a coordinated deploy. If all reconnect during the next second, the ingress sees 100,000 new handshakes per second plus authentication, subscription setup, and perhaps a snapshot query per client. If the same clients spread over 100 seconds, the average arrival rate is 1,000 per second. Both rates are derived as clients divided by seconds; neither predicts an actual fleet's capacity. The peak still depends on the distribution, retries, and work per reconnect. If a failed attempt schedules an immediate retry, the second wave adds to work that has not drained. A reconnect policy must protect downstream dependencies, not merely the socket accept loop.

A common client pattern is exponential backoff with jitter and a ceiling. In an illustrative implementation, delay attempt $n$ by a random value in $[0,\min(30\text{ s},2^n\text{ s})]$. That formula is an application policy example, not an RFC 6455 requirement. “Full jitter” prevents a cohort that disconnected at the same instant from choosing the same next instant again. But jitter alone does not solve overload if every reconnect causes an expensive full snapshot. Prefer resume from a cursor where possible, bound concurrent session creation, and let the server say when to retry if the application protocol supports that instruction. The [SRE retries and backoff post](/blog/software-development/site-reliability-engineering/timeouts-retries-and-backoff-done-right) covers retry policy in general; here the unit of work is an entire stateful connection plus catch-up.

Treat a deploy as a planned failure injection. Stop routing new connections to an old pod, allow existing sockets to drain for a finite period, and tell clients to resume elsewhere before the process exits if the protocol supports a meaningful close reason. A load balancer may still close a tunnel on pod termination, so the client must handle abrupt EOF. An application Close code is useful when it arrives, but it cannot be the only recovery signal. The reader should ask three operational questions: How many sockets were on the terminating node? How much state must each reconnect rebuild? Is that state already available on another node?


### Admission capacity is part of the reconnect budget

Model the reconnect front door as a queue. Let $A$ be reconnect attempts arriving per second and $C$ be the number of complete session resumptions the service can finish per second under the current CPU and dependency load. If $A>C$, the backlog grows by approximately $(A-C)t$ after $t$ seconds, before client retries add more work. This is an explanatory queue model, not a WebSocket equation. Suppose a drain creates 12,000 attempts over 10 seconds, or an average $A=1{,}200$/s, while the session service can safely complete $C=800$/s. The queue grows by about 4,000 attempts in that interval. If each waiting attempt consumes memory or a socket, a seemingly short drain can exceed capacity even when steady-state live connection count is comfortable. All values are illustrative inputs to the stated formula.

A retry rule that says “try again in one second” can then turn 4,000 waiting or failed attempts into another burst while the service is still working on the first wave. A server-side admission gate should reject or defer excess starts cheaply, and a client should treat such a response differently from an invalid credential or expired resume cursor. An application can precompute a resume ticket or keep session metadata small, but only if it can revoke it and reauthorize subscriptions appropriately. The cost of a reconnect is not only one TCP handshake: TLS validation, authentication, state load, subscriptions, replay, and fan-out registration may each cross a different capacity boundary.

For a planned drain, estimate the peak new-session rate from the maximum number of sockets on any node, the number of nodes drained together, and the chosen spread window. Observe the distribution, not just the mean. If 40 nodes each hold 10,000 connections and 4 drain together, 40,000 clients need a new route. Spreading their first attempts over 80 seconds produces a simple average of 500 attempts per second, before retries and naturally occurring connects. These are derived example values, not a recommendation to drain exactly four nodes or wait 80 seconds. The engineering decision is to pick a window the measured resume path can handle with margin, then test a zone-scale failure where the drain is not controlled.

Do not force an old process to hold sockets indefinitely just to avoid a reconnect spike. Then deployments accumulate stale connections, configuration, and security state. Budget reconnection capacity, make the transition observable, and bound how long a pod can drain. A graceful shutdown sequence might mark the node unready, stop accepting new upgrades, issue reconnect instructions or Close frames in batches, wait for a defined drain deadline, and terminate. The exact readiness and drain controls depend on the platform; the [health checks and graceful shutdown sibling](/blog/software-development/networking/health-checks-draining-and-the-graceful-shutdown-nobody-implements) carries the broader deployment race.

Metrics should make the reconnect wave visible before the application goes dark. Count new TCP accepts, successful upgrades, rejected upgrades by status, authentications, session resumes, full session starts, and current live connections separately. Watch reconnect attempts per client cohort and per source region, queue wait for admission, and downstream snapshot/read load. A single “connections” gauge can decrease while handshake attempts surge. If a deploy causes a step down followed by a jagged recovery, the rate and cost of reconnection matter more than the final steady-state count.

## 6. A public case: Discord's March 2026 reconnection cascade

[Discord Engineering's incident analysis, published April 29, 2026](https://discord.com/blog/behind-the-scenes-of-the-3-25-26-voice-outage), describes a useful case because it connects a stateful real-time service, WebSocket gateways, reconnection, and downstream overload in one documented chain. The event occurred March 25, 2026. Users mainly saw an “Awaiting Endpoint” state for voice and video, with major degradation from 12:13 to 15:30 PDT according to Discord. This was not a WebSocket framing bug. The socket layer carried and amplified the recovery work after a different trigger.

The trigger was a routine configuration change during Discord's Kubernetes migration. The team increased session-service pod resources and reduced pod count proportionally in its first zone. Kubernetes terminated half the pods in that zone. A handoff safety check held process transfer until other work finished; the termination grace period expired before handoffs could begin. Discord reports that 17% of sessions across its service stopped ungracefully. The source explains that a session is a stateful process associated with a connected device, and messages destined for a client pass through it. The number is Discord's scoped incident measurement, not a generic fraction of clients affected by every deploy.

The gateway is the ingress and egress for Discord's WebSocket traffic. It monitors user sessions and, on a session exit, tells the client to reconnect and tries to resume the session in the same zone. That strategy is sensible for an isolated failure. At 12:13, the strategy ran for a large simultaneous cohort. A per-host start-session rate limit that provided backpressure had not been retuned after the platform migration. As Discord explains, the higher pod count allowed more concurrent session starts through than intended. Gateway memory in the affected zone spiked, gateways restarted, and their other connected clients formed another reconnect wave. This is a second-order blast-radius multiplier: connections that survived the first session loss were pulled into the recovery load.

The resulting work reached voice infrastructure. Discord's report says voice syncers then encountered a separate connection-creation bottleneck in Erlang supervisor processes. Growing mailboxes delayed outgoing HTTPS connections and service-registration refresh, which further hindered recovery. A later full voice-syncer restart produced another cold-start herd and brief recovery before the system fell behind again. Discord eventually addressed the ingress of work through rate-limit tuning and cluster capacity, among other changes documented in the postmortem. The exact queue and fleet measurements in that report belong to Discord's architecture; they are not a performance promise for a different implementation.

The causal graph is the transferable part. A configuration change removed stateful sessions. The gateway's reconnect behavior made a large cohort do expensive work together. An outdated admission limit allowed too much concurrent work. A second gateway failure enlarged the cohort. Downstream connection creation and service discovery coupled recovery to another serial bottleneck. A team that monitors only WebSocket close rate or only voice RPC errors would see pieces of this story, not the cause of amplification. Correlate session exits, reconnect attempts, admission queue, gateway memory, and downstream connection start latency on one incident timeline.

This case also puts limits on our design advice. Jitter on clients can reduce synchronization, but it does not replace correct server-side admission control or safe state handoff. Durable replay can avoid full snapshots, but it cannot compensate for a service that cannot create any sessions. A graceful pod termination helps, but a sudden host failure still requires recovery. The control should be layered: planned draining, bounded reconnect admission, cheap resume, per-hop queue limits, and a way to shed or defer nonessential reconstruction while core service returns. The source owner, date, observed symptom, triggering change, network boundary, reported numbers, and transfer lesson are all explicit, so the case is evidence rather than an anecdote.

## 7. Fan-out that survives a process restart

A fan-out service reads one published event and sends it to many subscribed clients. The straightforward design keeps an in-memory map from topic to local sockets and pushes each event to every matching socket. That map is useful for routing on a live node, but it is not durable state. If the process restarts, the sockets and map disappear. If the publisher sent an event while the client was disconnected, no amount of reconnecting to a fresh node can reconstruct it from the old node's memory.

![Durable event log and client cursors separating replayable history from ephemeral socket membership](/imgs/blogs/websockets-sse-and-long-lived-connections-6.webp)

A resilient design separates three jobs. An ordered, durable event store retains the data for a defined period. A fan-out worker tracks which clients are currently attached to each node and writes live events to bounded per-client queues. Each client reports its last *applied* cursor when it reconnects; the server replays events after that cursor, then moves the client onto the live tail without a gap. The dangerous seam is between replay and live subscription. If a worker subscribes after it finishes replay, events published in the gap can be missed. If it subscribes before replay without deduplication, events can be sent twice. The design must choose an atomic log position, or otherwise overlap replay and live delivery with stable IDs and deduplication.

Here is an illustrative sequence. A topic log has committed events 501 through 506. The client applied 503 before its socket broke. It reconnects with cursor 503. The server replays 504, 505, and 506, then follows new committed events. If 506 is delivered twice around the handoff, the client ignores the second copy by ID. If the server's retention window now begins at 505, it cannot truthfully provide 504; it must direct the client to a snapshot and establish a new cursor from that snapshot. These IDs are an explanatory example, not reported traffic. The result depends on how the application's log, snapshot, and subscription transition are implemented.

The cursor should identify *applied* state, not merely a frame received by the browser's network stack. If the browser receives event 506 and crashes before updating its local model, resuming from 506 loses work. A local applied cursor can lag the last delivered event and safely cause a duplicate on reconnect if handlers are idempotent. If an event triggers an irreversible side effect, use a domain-specific transaction or idempotency key. Neither WebSocket nor SSE frames confer exactly-once side effects. A transport acknowledgement can establish one kind of receipt but not that a database commit occurred.


### A replay cursor needs a consistency point

There is a subtle race when a user connects to a busy topic. Suppose the replay worker reads through event 900 and then registers the socket for live publication. Event 901 can be committed between those two actions. If publication checks the subscriber map before registration, the new client will miss 901 forever. Reversing the order can deliver 901 through the live path before the replay worker reaches it. That is a duplicate rather than a gap, which is easier to repair if event IDs are stable and application handlers are idempotent. The event numbers here are illustrative positions in an ordered log, not measurements.

One implementation pattern is to register for live delivery, capture the log's current high-water mark, replay from the client's cursor through that mark, and then release buffered live events strictly after the mark. The registration and mark must have a defined ordering relative to commits. If the broker cannot give that ordering, use an overlapping range and deduplicate by ID. The buffer for live events arriving during replay must also be bounded. A very slow replay should not quietly collect unbounded fresh data while promising to catch up. A snapshot at a known log position can shorten the replay span, provided the snapshot and following events form one consistent state transition.

At the client, persist or retain the applied cursor with the state it describes. Updating the cursor before applying an event risks a gap after a crash. Applying an event before updating the cursor can cause a duplicate after a crash, which is acceptable only if the handler tolerates it. This is the same ordering problem that appears in message consumers, expressed through a browser connection. A transport protocol cannot decide the application's transaction boundary. The service must document whether IDs are ordered per topic, per user, or globally, and whether a cursor is valid after authorization changes.

Fan-out architecture also has a capacity shape. If an event of $S$ bytes goes to $F$ subscribers, the node or fleet must enqueue approximately $F\times S$ payload bytes before framing, TLS, and copies. This is a derived lower-order accounting model, not an exact memory measurement. For a 1 KiB event and 10,000 recipients, one publication represents about 9.77 MiB of payload deliveries. If that event arrives 20 times per second, the nominal payload delivery rate is about 195 MiB/s across all recipients. Network placement, compression, shared buffers, and filtering can change actual bytes and memory, but no protocol upgrade erases the fan-out work. Partition subscribers across nodes, reuse immutable payloads where safe, and avoid a single process becoming the only source of delivery history.

A broker or log is not automatically a replay guarantee. An ephemeral pub/sub bus can carry live events to many nodes yet discard anything published during a disconnect. A durable stream can retain records, but retention may expire; ordering may be per partition rather than global; a consumer offset may represent delivery or processing depending on the API. State the ordering scope and retention period in the application contract. If a user joins a room at a particular moment, define whether they should see history before joining. Authorization needs to be rechecked on reconnect and before replay; a valid cursor must not become a bypass to read events from a topic the user no longer has permission to access.

A deployment-friendly node needs little irreplaceable local state: active sockets, current subscriptions, and bounded output queues. A restart can lose those ephemeral objects, because clients can reconnect, reauthorize, and resume from the durable log. This is more honest than pretending connections can move between pods without interruption. L4 balancing still chooses a node for a TCP flow; a live WebSocket does not migrate because a different pod becomes ready. The [gRPC load balancing sibling](/blog/software-development/networking/grpc-on-the-wire-streams-deadlines-and-the-load-balancing-trap) describes the related distinction between connection-level and logical-request balancing. The higher-level service-discovery design belongs in [microservices service discovery](/blog/software-development/microservices/service-discovery-and-load-balancing).

## 8. Diagnose the failing boundary, not the protocol label

The first useful question in an incident is where the connection first stops being healthy. A browser close event is a downstream symptom. It may report an abnormal closure after a proxy reset, an app crash, an origin drain, or a laptop network change. The diagnosis needs one timeline per boundary: opening handshake status; last upstream byte; last downstream byte; Ping and Pong; per-client queue age; first transport close; and reconnect attempt. Align clocks or annotate their offsets before claiming that one event preceded another.

![Decision tree from failed upgrade, fixed idle interval, growing queue, or synchronized reconnects to the next measurement](/imgs/blogs/websockets-sse-and-long-lived-connections-7.webp)

If the handshake fails, inspect HTTP status and the hop that generated it. A `400` from the origin after a malformed key is different from a `403` due to authorization, and both differ from a proxy that never forwarded `Upgrade`. Record the response headers and compare upstream and downstream request IDs. A successful `101` at the edge but no corresponding origin `101` points to the proxy route or an intermediate policy. A `101` at both ends with later silence moves the investigation from handshake to tunnel lifetime. Do not keep retrying an authorization failure at network speed.

If closure comes after a nearly fixed quiet interval, look up the idle timer of every hop and observe which direction last carried bytes. An nginx upstream read timeout is explained by silence from upstream, even if the client sent browser messages. A client-side Ping, if the API supported it, would prove a different path from a server Ping. Capture near the proxy only when needed. Encrypted captures at the wrong side of TLS termination show TCP behavior but not WebSocket opcode; an application frame log at the terminating endpoint may answer the question more directly. The absence of a Close frame plus EOF or reset suggests transport loss or an intermediary action, while a received Close code points to an endpoint that participated in the protocol close.

If connection count remains steady but event latency grows, treat it as a queue problem. Inspect bytes queued per client, oldest event age, app publish-to-enqueue time, enqueue-to-write time, and browser receive-to-apply time. A slow subscriber can make one node's memory rise while the publish service remains fast. A proxy buffering SSE can make several events arrive in one burst even though the origin wrote each promptly. In both cases the visible symptom is “real-time feels late”; the fix depends on which queue owns the delay. Increasing a read timeout does nothing for an unbounded output queue.

If many connections drop together after a deploy, correlate the drop with pod termination and load-balancer draining rather than with a random network outage. Separate planned Close frames from resets. Count reconnect attempts per second, not merely successful established sockets. Compare resume rate to full reinitialization rate and downstream state reads. If the second reconnect wave follows gateway memory or admission queue growth, the system is amplifying its own recovery. Rate-limit session creation, spread planned drains, and make resuming cheaper before increasing gateway replica count blindly. The Discord case shows that the next bottleneck may be outside the gateway entirely.

A small production-facing set of read-only checks can narrow the boundary. `ss -tan state established '( sport = :8080 )'` shows a server's established TCP sockets on Linux, but it does not count subscriptions or prove frame delivery. `ss -tin` adds TCP diagnostics for selected sockets; retransmissions and send queues suggest transport or slow-reader pressure, yet an application queue can grow before `ss` sees bytes. Proxy access and error logs can show upgrade status and timeout reason if configured. App counters should report current socket count, Ping/Pong misses, queue bytes, queue age, and resume outcome. Always tag the same connection or session ID across layers, with privacy limits appropriate to your system.

Do not use `ping` to declare a WebSocket route healthy. ICMP reachability does not test the HTTP upgrade, TLS validation, proxy headers, origin authorization, or long-lived delivery. A synthetically opened socket that closes immediately tests only the opening path. A meaningful probe stays connected beyond the shortest configured idle timer, observes at least one heartbeat or application event, and tests reconnection from a cursor. Even that probe cannot stand in for slow subscribers, so exercise bounded queue policy separately.

## 9. Choose the protocol with the failure mode in mind

A protocol choice table can be brief because the causal sections above do the work. The key is to pick the simplest stream whose failure behavior your service can operate. None of these choices eliminates the need for authentication, proxy configuration, bounded queues, observability, and recovery semantics.

| Workload | Reasonable starting point | Operational condition | Source |
| --- | --- | --- | --- |
| Infrequent status changes with seconds of latency tolerance | Polling | Cache the response and bound request fan-out; simple reconnect is a new request | Derived design guidance |
| Server-originated feed or job progress | SSE | Flush records through proxies, retain IDs for useful replay, and bound slow readers | [WHATWG SSE](https://html.spec.whatwg.org/multipage/server-sent-events.html) |
| Frequent interactive messages in both directions | WebSocket | Define heartbeats, application resume, bounded queues, and deploy draining | [RFC 6455](https://www.rfc-editor.org/rfc/rfc6455) |
| Browser sends rare commands while receiving frequent updates | SSE plus ordinary HTTP requests | Keep command semantics independent from the event stream | Derived design guidance |

Do not infer that WebSocket is faster in every environment just because it avoids repeated polling requests. At low update frequency, an open connection can cost more memory and operational complexity than occasional HTTP requests. At high frequency, polling can waste request headers and add update latency tied to the polling interval. The break-even depends on request overhead, event rate, fan-out, proxy behavior, and how much state each reconnect rebuilds. Measure with the actual client mix. The number of bytes per message alone is not a complete cost model.

A good review asks the product owner what “lost update” means. For transient presence, a fresh snapshot may be enough. For notifications, a retained log and cursor may be needed. For commands, the browser should receive a separate operation acknowledgment or query the operation state after reconnect. If bidirectional commands are rare and the stream is only server-to-client, SSE's browser reconnection contract is useful. If the app really needs both peers to speak at arbitrary times over the same session, WebSocket is a natural fit, but the application still designs ordering, replay, and idempotency. Think through those guarantees before arguing about opcodes.

## Run it yourself

### Question

Can a proxy close a successfully upgraded connection because the upstream is silent, and can upstream heartbeat bytes prevent that specific idle close? This local experiment uses a tiny standard-library relay with an nginx-like *upstream-read* timer. It demonstrates the causal timer behavior, not nginx's complete implementation. The [series setup post](/blog/software-development/networking/what-actually-happens-when-you-curl-a-url) explains the canonical Linux `netlab` environment; this focused experiment binds only ephemeral loopback ports so it also runs on macOS without privileges.

### Preconditions

Use Python 3.10 or later on Linux or macOS, with permission to bind loopback sockets. No root, namespace, `tc`, capture, cloud account, or external dependency is needed. Run `python3 --version` first; this script was checked with Python 3.14.5 on September 30, 2026. It creates two ephemeral listeners on `127.0.0.1` and exits after each run. Do not paste the demonstration's fixed sample key into a production WebSocket client. The lab has no credentials or personal payloads.

The upstream sends a valid HTTP/1.1 `101` response and then either stays silent or sends an empty server-to-client Ping frame (`0x89 0x00`) once per second. A relay forwards bytes and closes when it has read nothing from upstream for 3 seconds. The client waits up to 5 seconds after the upgrade. The *only treatment variable* is the heartbeat interval: zero means no heartbeat, one means a Ping every second. These shortened timers make the result quick; they are not recommended production values.

### Baseline

Copy the complete shell block below. It writes only `/tmp/ws-idle-lab.py`; the script prints the effective configuration before it opens the client, so the measured duration can be interpreted.

```bash
cat > /tmp/ws-idle-lab.py <<'PY'
#!/usr/bin/env python3
"""Local WebSocket-like upstream through an upstream-read idle relay."""
import argparse
import base64
import hashlib
import select
import socket
import threading
import time

GUID = "258EAFA5-E914-47DA-95CA-C5AB0DC85B11"


def listen():
    listener = socket.socket()
    listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    listener.bind(("127.0.0.1", 0))
    listener.listen(1)
    return listener


def handshake(sock):
    request = b""
    while b"\r\n\r\n" not in request:
        request += sock.recv(4096)
    key_line = next(line for line in request.split(b"\r\n")
                    if line.lower().startswith(b"sec-websocket-key:"))
    key = key_line.split(b":", 1)[1].strip()
    accept = base64.b64encode(hashlib.sha1(key + GUID.encode()).digest())
    sock.sendall(b"HTTP/1.1 101 Switching Protocols\r\n"
                 b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                 b"Sec-WebSocket-Accept: " + accept + b"\r\n\r\n")


def upstream(listener, heartbeat):
    sock, _ = listener.accept()
    with sock:
        handshake(sock)
        if heartbeat == 0:
            time.sleep(7)
        else:
            end = time.monotonic() + 7
            while time.monotonic() < end:
                time.sleep(heartbeat)
                try:
                    sock.sendall(b"\x89\x00")  # RFC 6455 server Ping, empty payload
                except OSError:
                    break


def relay(listener, upstream_port, idle_seconds):
    client, _ = listener.accept()
    server = socket.create_connection(("127.0.0.1", upstream_port))
    with client, server:
        last_upstream_byte = time.monotonic()
        while True:
            remaining = idle_seconds - (time.monotonic() - last_upstream_byte)
            if remaining <= 0:
                print("relay_close=upstream_read_idle", flush=True)
                return
            readable, _, _ = select.select([client, server], [], [], remaining)
            for source in readable:
                data = source.recv(4096)
                if not data:
                    return
                target = server if source is client else client
                target.sendall(data)
                if source is server:
                    last_upstream_byte = time.monotonic()


def client(port, duration):
    sock = socket.create_connection(("127.0.0.1", port))
    with sock:
        key = base64.b64encode(b"0123456789abcdef")
        sock.sendall(b"GET /updates HTTP/1.1\r\nHost: localhost\r\n"
                     b"Upgrade: websocket\r\nConnection: Upgrade\r\n"
                     b"Sec-WebSocket-Key: " + key + b"\r\n"
                     b"Sec-WebSocket-Version: 13\r\n\r\n")
        response = b""
        while b"\r\n\r\n" not in response:
            response += sock.recv(4096)
        assert response.startswith(b"HTTP/1.1 101")
        print("upgrade=101", flush=True)
        started = time.monotonic()
        ping_count = 0
        sock.settimeout(0.2)
        while time.monotonic() - started < duration:
            try:
                data = sock.recv(4096)
            except socket.timeout:
                continue
            if not data:
                print(f"client_end=eof elapsed_s={time.monotonic()-started:.1f} pings={ping_count}")
                return
            ping_count += data.count(b"\x89\x00")
        print(f"client_end=duration elapsed_s={time.monotonic()-started:.1f} pings={ping_count}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--heartbeat", type=float, choices=[0, 1], required=True)
    args = parser.parse_args()
    upstream_listener = listen()
    relay_listener = listen()
    upstream_port = upstream_listener.getsockname()[1]
    relay_port = relay_listener.getsockname()[1]
    threads = [threading.Thread(target=upstream, args=(upstream_listener, args.heartbeat), daemon=True),
               threading.Thread(target=relay, args=(relay_listener, upstream_port, 3), daemon=True)]
    for thread in threads:
        thread.start()
    print(f"idle_s=3 heartbeat_s={args.heartbeat:g}", flush=True)
    client(relay_port, 5)
    for listener in (upstream_listener, relay_listener):
        listener.close()


if __name__ == "__main__":
    main()
PY
python3 --version
python3 /tmp/ws-idle-lab.py --heartbeat 0
```

Read `upgrade`, `relay_close`, `client_end`, `elapsed_s`, and `pings`. The expected qualitative state is `upgrade=101`, then `relay_close=upstream_read_idle`, and `client_end=eof` after roughly 3 seconds with zero Ping frames. On the tested host it printed 3.0 seconds. Expect approximately 2.8–3.5 seconds on a normally scheduled laptop; a heavily loaded machine can run late. A successful `101` followed by EOF proves that opening and lifetime are separate observations.

### Apply one change

```bash
python3 /tmp/ws-idle-lab.py --heartbeat 1
```

Now only the upstream's send interval changes. The relay's 3-second idle policy and the client's 5-second observation window stay fixed. Each Ping frame resets the relay's upstream-read clock. No client application data is sent, so we are testing the direction that the relay actually watches.

### Compare

```bash
python3 /tmp/ws-idle-lab.py --heartbeat 0
python3 /tmp/ws-idle-lab.py --heartbeat 1
```

Read the same fields from both runs. The baseline should end as `client_end=eof` around 3 seconds with `pings=0`. The treatment should end as `client_end=duration` around 5 seconds with about 4–5 Ping frames and no `relay_close=upstream_read_idle` during that window. On Python 3.14.5 in this workspace, the treatment printed 5.0 seconds and five Pings. Scheduler timing can change the count by one. This comparison establishes a local causal claim: bytes from upstream reset an upstream-read idle timer. It does not establish that a Ping reaches an application handler or that every proxy uses the same timeout definition.

### Reset

```bash
rm -f /tmp/ws-idle-lab.py
```

The processes and ephemeral loopback ports exit when each run ends; the reset removes only this lab's file. On a real Linux host, a safer read-only translation is to inspect the proxy's effective timeout and logs, then compare last upstream byte and disconnect timestamps. Do not modify an unspecified production proxy to reproduce a 3-second timeout. If capturing real traffic, use a narrow filter and protect payloads, cookies, and tokens.

## Key takeaways

A WebSocket upgrade proves that the opening HTTP exchange worked. The connection remains subject to each intermediary's idle policy, the endpoint's event loop, bounded output queues, and a recovery path for when it closes. Measure the direction of the last byte before changing heartbeat intervals. A `1006` observed in a browser is evidence of abnormal closure, not the cause.

SSE gives a strong server-to-client default when the application does not need bidirectional messages. Its event IDs and `Last-Event-ID` make a resume request straightforward, but the service still has to retain data and replay it correctly. Proxy buffering, idle timeouts, and slow subscribers remain operational concerns. Polling remains a sensible baseline when freshness requirements are loose.

Backpressure and reconnection are the two capacity cliffs. Bound each client's queue in bytes and age; define what to coalesce, drop, or replay. Spread reconnect attempts, but also cap server-side session creation and make resume cheaper than a full rebuild. Discord's March 2026 incident shows how an apparently local session loss can turn into a gateway reconnect wave and then overload a downstream dependency.

The operational test is simple to state and hard to fake: kill a fan-out node while publishing numbered events, reconnect the browser to a different node, and compare the set of applied IDs against the retained log. Repeat with a subscriber that reads slowly and with a cursor older than retention. A green upgrade metric is not a substitute for those three observations. The robust fan-out boundary is a durable sequence plus an ephemeral socket membership map. A restart may drop sockets, but a client with an applied cursor can reconnect to any healthy node, reauthorize, replay the gap, and rejoin the live tail. If retention no longer covers the gap, the server must say so and provide a snapshot path. That contract matters more than whether the live hop uses WebSocket or SSE.

## Further reading

- [RFC 6455: The WebSocket Protocol](https://www.rfc-editor.org/rfc/rfc6455), December 2011. Handshake, framing, control frames, masking, and closing behavior.
- [RFC 8441: Bootstrapping WebSockets with HTTP/2](https://www.rfc-editor.org/rfc/rfc8441), September 2018. Extended `CONNECT` instead of a classic HTTP/1.1 upgrade on capable paths.
- [WHATWG HTML: Server-sent events](https://html.spec.whatwg.org/multipage/server-sent-events.html). Event stream parsing, IDs, and reconnection behavior.
- [nginx: WebSocket proxying](https://nginx.org/en/docs/http/websocket.html). Upgrade forwarding and the upstream read timeout.
- [Discord Engineering: Behind the scenes of the March 25, 2026 voice outage](https://discord.com/blog/behind-the-scenes-of-the-3-25-26-voice-outage), published April 29, 2026. A documented reconnect cascade and downstream recovery bottleneck.
