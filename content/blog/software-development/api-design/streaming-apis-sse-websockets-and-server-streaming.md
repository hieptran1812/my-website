---
title: "Streaming APIs: SSE, WebSockets, and Server Streaming"
date: "2026-06-20"
publishDate: "2026-06-20"
description: "When one request and one response is not enough, you need a continuous flow — this is the deep dive on long polling, Server-Sent Events, WebSockets, and gRPC streaming, with the wire formats, reconnection, backpressure, authentication on a long-lived connection, and how to scale stateful connections behind a pub-sub edge."
tags:
  [
    "api-design",
    "api",
    "streaming",
    "sse",
    "websockets",
    "grpc",
    "http",
    "real-time",
    "backpressure",
    "rest",
  ]
category: "software-development"
subcategory: "API Design"
author: "Hiep Tran"
featured: true
readTime: 56
image: "/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-1.png"
---

The ops team at our fictional commerce platform had a dashboard problem. The finance lead wanted a screen that showed payouts moving through their lifecycle in real time — `pending`, then `processing`, then `paid`, then `settled` — so that when a partner called asking "where is my money," the answer was on the wall, not three clicks and a database query away. The first version was the obvious one: the browser called `GET /payouts?status=in_progress` every three seconds and re-rendered the table. It worked in the demo. Then we turned it on for forty operators across two offices, each tab polling every three seconds, and the payments API started seeing roughly **eight hundred requests a minute** for data that changed maybe a dozen times an hour. Ninety-nine percent of those requests returned the same rows they had returned three seconds earlier. We had built a very expensive way to learn that nothing had happened.

The fix was not a bigger database. It was a different *shape* of API. Instead of the client asking "anything new?" over and over, the server would hold a connection open and *push* each status change the instant it occurred. One request, many responses, flowing for as long as the dashboard stayed open. That is a **streaming API**, and once you have one in your toolbox you start seeing the cases for it everywhere: live prices on a trading screen, the progress bar on a long upload, a chat message arriving, a notification badge incrementing, the tokens of an LLM response appearing one word at a time, a log tail, a paginated export streamed row by row instead of buffered into one giant body. All of these share a property that the one-request-one-response model handles badly: the data is *produced over time*, and the client wants it as it arrives, not in one lump at the end and not by asking again and again.

This post is the practitioner's tour of the four ways to do that — **long polling**, **Server-Sent Events (SSE)**, **WebSockets**, and **gRPC streaming** — with the actual wire formats, worked examples you can copy, and the operational reality nobody mentions in the tutorial: how reconnection works, how you stop a slow consumer from running the server out of memory (this is **backpressure**, and it is the part people skip), how you authenticate a connection that lives for an hour, and how you scale tens of thousands of stateful connections across a fleet of servers behind a load balancer that wants to kill anything idle for sixty seconds. The frame is the same one that runs through this whole series: **an API is a contract, not a function call.** A streaming endpoint makes a *new kind* of promise to the caller — "I will keep this channel open, and here is exactly what flows down it, in what order, and what you should do when it drops" — and the quality of your streaming API is mostly the quality of how honestly you keep that promise when the network misbehaves.

![a comparison matrix of long polling, Server-Sent Events, WebSockets, and gRPC streaming across the four axes of direction, transport, browser support, and reconnection behavior](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-1.png)

If you have only ever written `@app.route("/payouts")` that returns a JSON list and exits, the mental shift is this: a streaming handler does *not* return. It opens a response and keeps writing to it. The function's lifetime is the connection's lifetime. Everything hard about streaming follows from that one fact — the connection is now a resource you own, hold, feed, and eventually have to clean up, and there can be a hundred thousand of them.

## When to stream (and when a normal request is fine)

Before the mechanisms, the judgment. Streaming is not free — it adds a stateful, long-lived connection to a system that was otherwise stateless and easy to scale horizontally — so you reach for it only when the access pattern genuinely needs it. The honest test is: **does the same client want multiple pieces of data over time from one logical operation, and does freshness matter on the order of seconds or better?** If yes, stream. If the client wants one snapshot and is happy to ask again later, a plain `GET` with good caching (covered in the caching post of this series) is simpler and cheaper.

Here are the patterns where streaming earns its keep, and what specifically makes a normal request fall short:

- **Live prices and market data.** A trading screen needs ticks as they print. Polling at a fixed interval is both too slow (you miss intra-interval moves) and too fast (most polls return no change). The data is a continuous firehose; the client wants the firehose.
- **Progress updates on slow work.** An import of fifty thousand orders takes two minutes. The client wants a progress bar, not a spinner. A stream of `progress: 40%` events turns a black box into a status bar. (When the work is *very* long and the client may disconnect, you combine streaming with the operation-resource pattern from the long-running-operations post — stream the progress, but also expose a pollable status so a reconnecting client can catch up.)
- **Chat and collaborative editing.** Messages arrive when *other people* act, not when this client asks. There is no request that "causes" the next message. This is the textbook case for a push channel — and because the client also sends, it is the textbook case for a *bidirectional* one.
- **Notifications.** A badge that should increment the moment something happens. Polling adds latency equal to half the poll interval on average and load equal to the client count divided by the interval; pushing adds neither.
- **LLM token streams.** A model generates text token by token over several seconds. Buffering the whole completion and returning it at the end means the user stares at nothing for five seconds; streaming the tokens means the answer starts appearing in a few hundred milliseconds. This is why essentially every chat-model API streams.
- **Live dashboards.** Our payouts wall — many clients watching the same slowly-changing state, each wanting updates promptly.
- **Large result sets.** An export of ten million rows. Buffering them into one JSON array means the server holds the whole thing in memory and the client waits for all of it before seeing the first row. Streaming the rows (server streaming, or newline-delimited JSON over a plain response) bounds server memory and lets the client process row one while row two is still being computed.

And the cases where you should *not* stream, because a simpler design covers you:

- **A one-shot read with a snapshot answer.** `GET /orders/123`. The client wants the order as it is now. Add an `ETag` and let the client conditionally re-fetch. No stream.
- **Infrequent, non-urgent updates.** Something that changes a few times a day and where a minute of staleness is fine. A poll every minute is trivial and robust; a held-open connection per client is overkill.
- **Fire-and-forget commands where the client does not care to watch.** Just `POST` it, return `202 Accepted`, done.
- **Anywhere the infrastructure actively fights long connections** and you cannot fix it — some legacy corporate proxies, some serverless platforms with short execution caps. If you cannot hold a connection open reliably, long polling (which uses short-lived requests) may be the only thing that survives.

There is also a subtler distinction hiding in this list that decides your wire format later: **event semantics versus state semantics.** Some streams carry *events* — discrete things that happened, each meaningful on its own, none superseding another: a payout transitioned, a message was posted, an order filled. Losing one is losing information. Other streams carry *state* — the current value of something, where only the latest matters and every update supersedes the one before: a price, a progress percentage, a "users online" count. Losing an intermediate state value is fine because the next one replaces it anyway. This distinction will come back hard in the backpressure section, because it is what tells you whether you may drop data under load (state: yes, drop the stale one) or must never drop it (events: no, you must persist and resume). Decide which kind each of your streams is *now*, while you are designing it, not later when the server is on fire.

The decision is not "streaming is modern, polling is old." It is a contract question: *what does the caller need to assume about freshness, and how many of them are there?* Keep that in mind as we walk each option.

## Long polling: the baseline everything is measured against

Start with the simplest thing that pushes data, because it sets the bar and because it is the fallback when nothing else works. **Long polling** is ordinary HTTP with one twist: instead of the server answering immediately, it *holds the request open* until either new data is available or a timeout fires, then responds. The client, on receiving the response, immediately makes another request. The connection is short-lived (one request, one response, like always), but the *experience* approximates a push because the server only answers when there is something to say.

Contrast it with **short polling**, the naive version from our dashboard's first draft: the client requests on a fixed timer regardless of whether anything changed. Short polling's cost is constant and mostly wasted; long polling's cost is proportional to the rate of actual events plus a trickle of timeout-driven re-requests.

Here is a long-poll request and the two ways it can resolve. The client passes a cursor — the id of the last event it saw — so the server knows where to resume:

```http
GET /payouts/po_88c1/events?after=1042 HTTP/1.1
Host: api.shop.example
Authorization: Bearer <token>
Accept: application/json
```

If an event exists after cursor `1042` within the server's wait window (say 25 seconds), the server answers right away:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store

{
  "events": [
    { "id": 1043, "type": "payout.processing", "at": "2026-06-20T10:01:04Z" }
  ],
  "next_cursor": 1043
}
```

If nothing happens before the wait window expires, the server returns an empty result (or a `204 No Content`) and the client re-requests with the same cursor:

```http
HTTP/1.1 200 OK
Content-Type: application/json
Cache-Control: no-store

{ "events": [], "next_cursor": 1042 }
```

The wait window matters and is a real design decision. Hold too long and you collide with intermediary timeouts (browsers, proxies, and load balancers will cut an idle-looking connection — often around 30 to 60 seconds); hold too short and you are back toward short polling's waste. A common choice is 25 to 30 seconds, comfortably under the typical 60-second proxy idle cap.

#### Worked example: why long polling is the robust fallback but not the default

Consider the payouts dashboard with forty operators. With **short polling** at three seconds, that is $40 \times (60 / 3) = 800$ requests per minute, essentially all of them empty. With **long polling** at a 25-second window and a real event rate of, say, twelve changes per hour, each client makes roughly one timeout-driven re-request every 25 seconds — about $40 \times (60 / 25) \approx 96$ requests per minute — plus the handful that carry actual events. That is an order of magnitude fewer requests, and crucially the *useful* events arrive within milliseconds instead of within up to three seconds.

So why isn't long polling the answer for everything? Two reasons. First, it still pays the full HTTP request overhead — headers, TLS resumption, a new server-side handler invocation — on every cycle, including every timeout. Second, and more importantly, the server is holding a request thread or task open while it waits, and on a thread-per-request server that means a held thread per waiting client. A hundred thousand long-poll waiters can exhaust a thread pool that a hundred thousand SSE streams on an async server would handle comfortably. Long polling is the universal fallback — it works through *anything* that speaks HTTP, including hostile proxies — but it is a fallback, not a destination.

| | Short polling | Long polling |
| --- | --- | --- |
| Latency to new data | up to the interval | near-immediate |
| Wasted requests | most of them | only timeout re-requests |
| Server resource while idle | none between polls | a held request per waiter |
| Works through hostile proxies | yes | yes |
| Complexity | trivial | low |

The takeaway: long polling is the floor. If a fancier transport is blocked by your infrastructure, long polling will still work. But when you *can* hold a real persistent channel, the rest of this post is better.

One more consequence worth spelling out, because it is a classic production incident. Long polling has a subtle race: between the moment the server answers a poll and the moment the client issues its next poll, there is a window — small, but real — during which the server is not holding an open request for that client. If an event fires in that gap and the server only pushes to *currently-connected* waiters, the client misses it. This is exactly why the cursor in the request matters: the client says `?after=1043`, and the server returns *everything* after 1043 whether or not the client was connected when it happened. Drop the cursor and "optimize" by only delivering live events to connected waiters, and you have built a stream that loses any event that lands during the reconnect gap — invisibly, intermittently, and impossible to reproduce on a developer's fast local loop. The cursor is not an optimization; it is what makes long polling *correct*.

## Server-Sent Events: the one-way workhorse

If your data flows **server to client only** — prices, progress, notifications, status changes, LLM tokens — and the client is a browser or anything that speaks HTTP, **Server-Sent Events** is very often the right answer, and it is criminally underused because people reach for WebSockets out of habit. SSE is a one-way, server-to-client stream carried over a perfectly ordinary HTTP response with the media type `text/event-stream`. There is no protocol upgrade, no special port, no new handshake — it is a `GET` that never finishes, dribbling out text in a tiny, well-specified format. Because it is plain HTTP, it sails through proxies, firewalls, and HTTP/2 multiplexing that choke on exotic protocols, and the browser gives you automatic reconnection for free.

![a before and after contrast showing short polling spending most requests on empty responses versus a single SSE connection pushing only the events that changed](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-2.png)

### The wire format

The SSE protocol is almost embarrassingly simple. The server responds with `Content-Type: text/event-stream` and then writes UTF-8 text composed of `field: value` lines. A blank line terminates one *event* and dispatches it to the client. The fields you care about are:

- `data:` — the payload. Multiple `data:` lines in one event are concatenated with newlines, which is how you stream a multi-line message.
- `event:` — an optional event *name*, so the client can route different kinds of events to different handlers (e.g. `payout.paid` versus `payout.failed`).
- `id:` — an optional event id. The browser remembers the last id it received and replays it on reconnect via the `Last-Event-ID` request header. This is the resume mechanism, and it is built in.
- `retry:` — an optional integer in milliseconds telling the client how long to wait before reconnecting after a drop.
- A line beginning with `:` is a comment, ignored by the client — perfect for a heartbeat that keeps the connection from being reaped by an idle proxy.

#### Worked example: an SSE endpoint for payout status

Here is the server side. Notice the handler does not return — it loops, writing events as they arrive from an internal source (a message-queue subscription, a database change feed, whatever produces payout events). This is Python with a generator, but the shape is identical in Node, Go, or anything else:

```python
from flask import Response, request, stream_with_context

@app.route("/payouts/<payout_id>/stream")
def stream_payout(payout_id):
    last_seen = request.headers.get("Last-Event-ID")

    @stream_with_context
    def event_stream():
        # tell the client to wait 3s before reconnecting on a drop
        yield "retry: 3000\n\n"
        for ev in subscribe_payout_events(payout_id, after=last_seen):
            yield f"id: {ev.id}\n"
            yield f"event: {ev.type}\n"
            yield f"data: {json.dumps(ev.body)}\n\n"
            # heartbeat comment if the source goes quiet (handled by the source)

    return Response(
        event_stream(),
        mimetype="text/event-stream",
        headers={
            "Cache-Control": "no-store",
            "X-Accel-Buffering": "no",   # tell nginx not to buffer the stream
        },
    )
```

Two operational headers there matter more than they look. `Cache-Control: no-store` keeps any cache from holding the open stream. `X-Accel-Buffering: no` tells nginx (a very common reverse proxy) not to buffer the response — without it, nginx may collect your events and deliver them in a clump, defeating the whole point. If you run behind nginx and your SSE arrives in bursts, this header is almost always why.

Now the wire. When the client connects, the response begins and never closes; events stream as `text/event-stream`:

```http
GET /payouts/po_88c1/stream HTTP/1.1
Host: api.shop.example
Accept: text/event-stream
Authorization: Bearer <token>

HTTP/1.1 200 OK
Content-Type: text/event-stream; charset=utf-8
Cache-Control: no-store
Connection: keep-alive

retry: 3000

id: 1001
event: payout.processing
data: {"payout_id":"po_88c1","status":"processing","amount":"482.50"}

: heartbeat

id: 1002
event: payout.paid
data: {"payout_id":"po_88c1","status":"paid","paid_at":"2026-06-20T10:03:11Z"}

```

The `: heartbeat` comment line between events is the keepalive — it produces traffic on the connection so an idle proxy does not decide the stream is dead and close it. Send one every 15 to 30 seconds whenever the real event source is quiet.

### The client side is one constructor

In the browser, you do not parse any of this by hand. The `EventSource` API does it for you, including reconnection:

```javascript
const es = new EventSource("/payouts/po_88c1/stream");

es.addEventListener("payout.processing", (e) => {
  const body = JSON.parse(e.data);
  renderStatus(body); // "processing"
});

es.addEventListener("payout.paid", (e) => {
  const body = JSON.parse(e.data);
  renderStatus(body); // "paid" — flash it green
  es.close();         // we are done with this payout
});

es.onerror = () => {
  // EventSource is ALREADY retrying — this is informational.
  // It will reconnect automatically and send Last-Event-ID.
};
```

That is the whole client. When the connection drops, `EventSource` reconnects on its own after the `retry:` interval and sends `Last-Event-ID: 1002` so the server resumes from the next event. You did not write reconnection logic — the protocol and the browser handle it.

![a timeline of an SSE stream that drops mid-flight and resumes from the next event because the browser replays the last id it saw via the Last-Event-ID header](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-3.png)

### The principle: resumability is a contract, and the id makes it provable

Here is the part that elevates SSE from "neat trick" to "reliable contract." The promise a good stream makes is **at-least-once, gap-free delivery across reconnects** — if the connection drops, the client will receive every event it missed, in order, with nothing skipped. SSE makes that promise *checkable* with the `id:`/`Last-Event-ID` pair, but only if your server honors the contract on its side. The rule is: **the server must be able to resume from any id it has handed out**, which means the events need to be persisted (or at least buffered) long enough that a reconnecting client can catch up.

Why does this matter so much? Because the failure mode without it is silent. Suppose the connection drops right after the server sent `id: 1002` but before the client processed it, and on reconnect the server ignores `Last-Event-ID` and just starts from "now," sending `id: 1005`. The client never sees 1003 and 1004. Nothing errored. The dashboard just quietly missed two payouts going from `processing` to `paid`, and the first anyone hears of it is a partner asking why the wall says their money is still processing when it cleared an hour ago. A gap in a stream is a correctness bug that looks like nothing at all. So: hand out monotonic ids, persist enough history to replay them, and on every reconnect resume from `Last-Event-ID + 1`. That is the difference between a stream you can trust and a stream that lies by omission.

### What SSE cannot do

SSE is **one-way and text-only**. The client cannot send anything back over the same channel (it sends a separate ordinary HTTP request if it needs to). And the payload is UTF-8 text — you can ship JSON, but not raw binary without base64-encoding it (which inflates size by about a third). For a server-push feed of text or JSON, neither limit usually bites. The moment you need the client to push too — chat, a cursor position in a collaborative editor, a game input — you have outgrown SSE.

| Strength | Weakness |
| --- | --- |
| Plain HTTP — proxy and firewall friendly | Server-to-client only |
| Auto-reconnect with `Last-Event-ID` built in | Text only (binary needs base64) |
| Works over HTTP/1.1 and HTTP/2 | Per-domain connection limit on HTTP/1.1 |
| Trivial client (`EventSource`) | No native client-to-server channel |

One footnote on that "per-domain connection limit": over HTTP/1.1 browsers cap concurrent connections to a host at around six, and an open SSE stream eats one of those for its entire life. Six tabs each holding an SSE stream can starve the seventh. Over **HTTP/2** this evaporates — many streams multiplex over one connection — so if you lean on SSE heavily, serve it over HTTP/2.

There is a second, sneakier consequence of that limit that bites during development and rarely in the demo: if the SSE stream holds one of the six connection slots and your page also fires a handful of normal `fetch` calls to the same host, those calls can queue *behind* the held stream and appear mysteriously slow. The page is not slow because the server is slow; it is slow because the browser ran out of connections to that host and is waiting for one to free up — which the SSE stream never will. Again, HTTP/2 makes this vanish, which is one more reason the modern advice is simply "serve SSE over HTTP/2 and stop worrying about the connection budget." On HTTP/1.1, a common mitigation is to serve the stream from a dedicated subdomain so it does not compete with the page's regular requests for the same six slots.

## WebSockets: full-duplex when the client talks back

When the client must send *and* receive over the same live channel — chat, collaborative editing, multiplayer games, an interactive terminal, a trading UI that both shows prices and accepts orders on the same socket — you want **WebSockets**. A WebSocket is a single, long-lived TCP connection, established by *upgrading* an ordinary HTTP request, over which both ends can send messages at any time, in either direction, independently. That is what "full-duplex" means: not request-then-response, but two independent streams of messages sharing one pipe.

### The upgrade handshake

A WebSocket connection is born as an HTTP `GET` carrying special headers that ask the server to switch protocols. If the server agrees, it answers `101 Switching Protocols` and the connection stops speaking HTTP and starts speaking the WebSocket framing protocol (defined in RFC 6455). Here is the handshake on the wire:

```http
GET /ops/dashboard/ws HTTP/1.1
Host: api.shop.example
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Key: dGhlIHNhbXBsZSBub25jZQ==
Sec-WebSocket-Version: 13
Sec-WebSocket-Protocol: ops.v1
Authorization: Bearer <token>

HTTP/1.1 101 Switching Protocols
Upgrade: websocket
Connection: Upgrade
Sec-WebSocket-Accept: s3pPLMBiTxaQ9kYGzzhZRbK+xOo=
Sec-WebSocket-Protocol: ops.v1
```

A few things to read off that. `Sec-WebSocket-Key` is a random client nonce; the server hashes it with a fixed magic string and returns the result as `Sec-WebSocket-Accept`, which proves to the client that it reached a real WebSocket server and not a cache replaying an old response. `Sec-WebSocket-Version: 13` is the only version anyone uses. And `Sec-WebSocket-Protocol: ops.v1` is a **subprotocol** — an application-level protocol name both sides agree on, your hook for versioning the message format that flows over the socket (more on that under evolution below).

![a timeline of the WebSocket upgrade handshake turning an ordinary HTTP request into a persistent full-duplex channel after a 101 Switching Protocols response](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-7.png)

### Frames and messages

After `101`, data travels as **frames**. A frame has a small header (a few bytes) plus a payload, and frames carry either text (UTF-8) or binary, plus control frames for `ping`, `pong`, and `close`. Messages can be split across multiple frames (fragmentation), so a large message does not have to be buffered whole before the first byte goes out. You almost never touch frames directly — a library hands you whole messages — but it helps to know that the per-message overhead is tiny (single-digit bytes), which is why WebSockets suit high-frequency small messages where HTTP's header overhead per message would dominate.

#### Worked example: a live ops dashboard over WebSocket

The dashboard both *receives* payout updates and *sends* commands — for instance, an operator clicking "subscribe to high-value payouts only." Same socket, both directions. Browser client:

```javascript
const ws = new WebSocket("wss://api.shop.example/ops/dashboard/ws", "ops.v1");

ws.onopen = () => {
  // client -> server: refine what this dashboard wants
  ws.send(JSON.stringify({ type: "subscribe", filter: { min_amount: 1000 } }));
};

ws.onmessage = (frame) => {
  const msg = JSON.parse(frame.data);
  if (msg.type === "payout.update") updateRow(msg.payout);
  if (msg.type === "pong") noteHeartbeat();
};

ws.onclose = (e) => {
  // NOT automatic — unlike EventSource, you reconnect yourself.
  scheduleReconnect(e.code); // with backoff; resume from a cursor we kept
};
```

And the message exchange on the wire, shown as the logical messages flowing each way after the handshake:

```json
// client -> server
{ "type": "subscribe", "filter": { "min_amount": 1000 } }

// server -> client
{ "type": "subscribe.ok", "cursor": "evt_5567" }

// server -> client (later, when a payout changes)
{ "type": "payout.update", "cursor": "evt_5568",
  "payout": { "id": "po_91a4", "status": "paid", "amount": "2150.00" } }

// client -> server (keepalive request, app-level)
{ "type": "ping" }

// server -> client
{ "type": "pong" }
```

Notice two design choices that make this a *contract*, not just a pipe. First, every message has a `type` field, so the receiver can dispatch — this is your message schema, and it deserves the same care as a REST body. Second, server messages carry a `cursor`, so if the socket drops and the client reconnects, it can send `{ "type": "resume", "cursor": "evt_5568" }` and the server replays from there. WebSockets give you *no resume mechanism of their own* — there is no `Last-Event-ID` equivalent — so if you want gap-free delivery across reconnects (and on a payments dashboard you do), you build the cursor into your message protocol yourself. This is the single biggest thing people forget when they choose WebSockets: you inherit none of SSE's free reconnection-and-resume; you must design it.

### The cost of bidirectionality

A WebSocket buys you the client-to-server channel, but you pay for it:

- **No automatic reconnection.** The browser's `WebSocket` object does *not* reconnect on its own. You write the backoff loop, and you write the resume-from-cursor logic.
- **No HTTP semantics on the messages.** Inside the socket there are no status codes, no caching, no `Retry-After`, no content negotiation. You re-invent whatever of that you need as application messages. The disciplined REST contract you spent the rest of this series building does not automatically apply inside the frame stream.
- **Proxy and infrastructure friction.** Because it is not plain HTTP after the upgrade, some older proxies and corporate firewalls block or mangle WebSocket connections. It is far better than it was, but more fragile than SSE.
- **Stateful and harder to scale.** The connection is pinned to one server for its whole life, which complicates load balancing and deploys (you cannot drain a server without disconnecting its sockets). We tackle this in the scaling section.

The rule of thumb: **use a WebSocket only when you genuinely need client-to-server messages on the same live channel.** If the client only *receives*, SSE gives you the same push with free reconnection and less infrastructure risk. People reach for WebSockets reflexively for one-way feeds and then have to hand-build the reconnection that SSE would have given them for free. Direction first; everything else follows.

### Evolving the message contract on a long connection

The rest of this series spends a lot of energy on safe evolution — adding a field without breaking a client, never removing a response field, the tolerant-reader principle. Those rules do not evaporate inside a stream; they get *harder*, because there is no per-request `Accept` header to negotiate a version on every message and a connection may stay open across a deploy that changed the message format. So you have to design versioning into the stream itself.

The first hook is the one we already saw: the WebSocket **subprotocol**. When the client requests `Sec-WebSocket-Protocol: ops.v1` and the server echoes it, both sides have agreed on a message format for the life of the connection. To ship `ops.v2`, you teach the server to accept both, the client requests `ops.v2`, and old clients keep getting `ops.v1` — a clean, connection-scoped version negotiation with no flag-day. For SSE there is no subprotocol, so you version in the URL path (`/v2/payouts/stream`) or carry a `version` field in each event's `data`. For gRPC the `.proto` package (`payments.v1`) is the version, and Protobuf's own additive-change rules apply field by field.

Within a chosen version, the *message-level* compatibility rules are the same robustness-principle rules as for REST bodies, and they bind both directions:

- **Adding an optional field to a server message is safe** — a tolerant client ignores fields it does not recognize. A WebSocket or SSE client must be written to ignore unknown message `type`s and unknown fields, exactly as a REST client ignores unknown JSON keys. If your client `switch`es on `type` with no `default` branch that quietly ignores, you have built an intolerant reader that will break the day you add an event type.
- **Adding a new server-to-client message type is safe only if old clients ignore it.** New `payout.held` event? Old dashboards that do not know `held` must not crash or render garbage — they should drop it. Test this explicitly.
- **Removing or renaming a field or message type is breaking**, same as in REST. A client mid-stream that suddenly stops getting the `amount` field it depends on fails silently. Deprecate first (keep emitting the old field alongside the new), then remove only after clients have migrated — and on a streaming surface you often cannot tell who is still on the old shape, so lean conservative.
- **Changing the meaning of a field without renaming it is the worst kind of break** — the client keeps reading `status` and silently misinterprets it. Never repurpose a field; add a new one.

The practical upshot: a streaming client must be a **tolerant reader** from day one — ignore unknown message types, ignore unknown fields, never assume the set of event types is closed — because you will add to the stream over years and you cannot force every long-lived connection to reconnect on your schedule. A connection opened before your deploy is, in effect, pinned to the contract that existed when it opened, until it reconnects. Design as if every message format must coexist with the one before it.

## gRPC streaming: the internal, polyglot option

The fourth option lives mostly *inside* your system, between services, rather than out to browsers. **gRPC** (covered in depth in the gРС and Protocol Buffers post of this series) runs over HTTP/2 and supports four call shapes, three of which stream:

- **Unary** — one request, one response (the familiar RPC).
- **Server streaming** — one request, a *stream* of responses. The server-push analogue: ask once, receive many. This is the natural fit for "give me all payout events for this account" returning a flow of messages.
- **Client streaming** — a stream of requests, one response. Upload a flow of records, get one summary back.
- **Bidirectional streaming** — independent streams both ways over one call, the full-duplex analogue of WebSockets, but with a typed contract.

The big wins are the typed contract and codegen. You declare the stream in a `.proto` file and generate strongly-typed client and server stubs in any supported language, so a Go service and a Python service speak the same stream with no hand-written framing:

```protobuf
syntax = "proto3";
package payments.v1;

service Payouts {
  // server streaming: ask once, receive a flow of status events
  rpc WatchPayout(WatchPayoutRequest) returns (stream PayoutEvent);
}

message WatchPayoutRequest {
  string payout_id = 1;
  string resume_from = 2;   // a cursor for gap-free resume
}

message PayoutEvent {
  string id = 1;
  string type = 2;          // "processing", "paid", "settled"
  string status = 3;
  string occurred_at = 4;
}
```

gRPC streaming also gives you **deadlines** (the caller sets a maximum time for the whole call, propagated through the stack), typed status codes, and flow control built on HTTP/2's own windowing — which, as we will see, means backpressure is partly handled *for* you at the transport layer. The catch for browser-facing APIs is real: browsers cannot speak raw gRPC, so reaching a browser requires **gRPC-Web** plus a proxy (Envoy is the usual one) to translate, and even then full bidirectional streaming is limited. That is why the honest recommendation is: **gRPC streaming for internal and polyglot service-to-service streaming; SSE or WebSocket at the browser edge.** Often you combine them — a gRPC server stream inside the fleet, terminated at the edge and re-emitted to the browser as SSE.

#### Worked example: a gRPC server stream with a deadline and a resume cursor

Here is the consumer side of that `WatchPayout` server-streaming RPC, in Python, showing the two things gRPC gives you that a hand-rolled stream does not: a **deadline** that bounds the whole call, and the typed `PayoutEvent` messages that arrive with no parsing on your part. The `resume_from` cursor in the request is what makes the stream gap-free across a reconnect — exactly the same resumability contract as SSE's `Last-Event-ID`, but expressed in your own `.proto`:

```python
import grpc
from payments.v1 import payments_pb2 as pb
from payments.v1 import payments_pb2_grpc as rpc

def watch(channel, payout_id, last_cursor):
    stub = rpc.PayoutsStub(channel)
    req = pb.WatchPayoutRequest(payout_id=payout_id, resume_from=last_cursor)
    # the deadline bounds the WHOLE stream; renew on reconnect
    try:
        for ev in stub.WatchPayout(req, timeout=300):  # 5 minutes
            apply_event(ev)            # typed PayoutEvent, no JSON parsing
            last_cursor = ev.id        # remember for resume
            if ev.type == "settled":
                break
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.DEADLINE_EXCEEDED:
            reconnect_and_resume(payout_id, last_cursor)  # normal, not an error
        else:
            raise
    return last_cursor
```

The deadline is the part people misuse. In a unary RPC a deadline is a timeout on a single response; in a *stream* it is a cap on the entire call's lifetime, after which the call ends with `DEADLINE_EXCEEDED` even if the stream was healthy. So for a long-lived watch you do not set the deadline to infinity (that hides a dead connection forever); you set a generous one — minutes — and treat its expiry as a *normal* reconnect-and-resume event, not an error, refreshing the deadline each time. That pattern (bounded deadline plus resume cursor) gives you the liveness of a heartbeat and the safety of an upper bound at once.

For **client streaming** — say, a service uploading a flow of reconciliation records and getting one summary back — the shape inverts: the client iterates `yield`-ing request messages and the server returns a single response when the client signals end-of-stream. For **bidirectional streaming**, both sides iterate independently, which is how you build, for example, a long-lived control channel between two internal services where each pushes to the other as events occur. In all three, the typed contract and codegen are the point: a Go publisher and a Python consumer share one `.proto` and never hand-write a framing byte.

## Choosing among them

Now the decision, with the forces laid out. The first and most decisive question is **direction**: does only the server send, or does the client send too? That single answer eliminates half the options.

![a decision tree for choosing a streaming transport where direction is the first split and browser reach and infrastructure decide the leaf](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-5.png)

Here is the full comparison across the axes that actually drive the choice:

| | Long polling | SSE | WebSocket | gRPC stream |
| --- | --- | --- | --- | --- |
| Direction | server to client | server to client | full-duplex both ways | uni or bidirectional |
| Transport | plain HTTP, short requests | plain HTTP, one long response | upgraded TCP after `101` | HTTP/2 frames |
| Browser support | universal | native `EventSource` | native `WebSocket` | needs gRPC-Web + proxy |
| Reconnect + resume | you re-request with a cursor | automatic via `Last-Event-ID` | you build it (cursor in messages) | deadline + retry, you resume |
| Backpressure | natural (client controls pace) | limited (TCP, app-level) | app-level, you manage | HTTP/2 flow control built in |
| Proxy and firewall friendliness | excellent | excellent | good, occasionally blocked | good internally |
| Message vs event semantics | request/response | named events with ids | arbitrary messages | typed messages |
| Best for | a robust fallback | live prices, progress, tokens, dashboards | chat, collab editing, games | internal service-to-service streams |

Read it as a sequence of cuts. **One-way and to a browser?** SSE, almost always — you get free reconnection and proxy-friendliness. **Two-way to a browser?** WebSocket — there is no real alternative for client-to-server-on-the-same-channel in a browser. **Service-to-service inside the fleet?** gRPC streaming — typed contract, codegen, deadlines, transport-level flow control. **Stuck behind hostile infrastructure that breaks everything fancy?** Long polling — it survives where the others die. The mistake to avoid is choosing by familiarity ("we always use WebSockets") rather than by force; you end up hand-building reconnection you could have had for free, or fighting a proxy that would have passed SSE without a murmur.

## Backpressure and flow control: the part everyone skips

This is the section that separates a streaming API that survives production from one that falls over on a Tuesday. The problem is simple to state and brutal in practice: **what happens when the server produces events faster than a client can consume them?** Maybe the client is on a slow mobile link, maybe its tab is backgrounded and throttled, maybe its CPU is pegged. The events keep coming from your source. Where do they go while they wait for that one slow client to catch up?

### Derive why an unbounded buffer kills you

The naive implementation queues unconsumed events in memory per connection: as events arrive, append to the client's buffer; as the socket drains, send and remove. If the producer rate $r_{\text{in}}$ exceeds the client's drain rate $r_{\text{out}}$, the buffer grows at $r_{\text{in}} - r_{\text{out}}$ events per second, without bound. The memory used by one slow client after $t$ seconds is approximately $(r_{\text{in}} - r_{\text{out}}) \times t \times s$, where $s$ is the bytes per event. Plug in numbers: a price feed at 500 events/sec, a stalled client draining 50/sec, 200 bytes per event — that is $(500 - 50) \times 200 = 90{,}000$ bytes per second, or about **5.4 MB per minute, for one client.** Multiply by a few thousand connected clients where even a small fraction stall, and you have an out-of-memory crash that takes down *every* connection on the box, including the healthy ones. The slow consumer does not just hurt itself; it kills the server for everyone. An unbounded buffer is not a buffer, it is a memory leak with extra steps.

So the buffer must be bounded, which forces the real question: **when the bounded buffer fills, what do you give up?** Because you cannot keep everything, and you cannot block forever. Every backpressure strategy is an answer to "what do we sacrifice," and the right answer depends on what your data *means*.

![a stack of backpressure strategies from the dangerous unbounded buffer through bounded buffers and dropping to sampling and disconnecting the slow client](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-6.png)

### The strategies, from cheapest to costliest

- **Block the producer.** When the buffer is full, stop reading from the source until the client drains. Safe and simple, but only viable when the source *can* be slowed — a database cursor you control, a file you read, a gRPC stream where HTTP/2 flow control naturally propagates the stall upstream. If the source is a shared firehose feeding many clients (a market-data feed), blocking it for one slow client starves all the others. Do not block a shared source.
- **Bounded buffer, drop on full.** Keep a fixed-size buffer; when it overflows, drop new events. Bounds memory hard. The cost is gaps — and whether a gap is acceptable depends entirely on the data. For a price feed, a dropped tick is fine; the next tick supersedes it. For a payout state change, a dropped event is a correctness bug. So drop only **superseding** data.
- **Drop oldest, keep newest.** For state where only the latest value matters (a price, a progress percentage, a current status), evict the oldest queued item and keep the freshest. The client that catches up sees the *current* state, not a stale backlog. This is "last-value-wins" and is exactly right for dashboards.
- **Sample or coalesce.** Instead of every event, send at most one per interval — collapse a burst of 50 price updates in 100 ms into a single "here is the current price" message. This bounds the *rate*, not just the buffer, and matches human perception (nobody can read 500 updates a second anyway). For our dashboard, coalescing per payout per 100 ms means a payout that flickers through three states in one tick arrives as its final state.
- **Disconnect the slow client.** When a client falls hopelessly behind, close its connection with a clear code and let it reconnect and resume from a cursor (catching up via a snapshot plus the events since). This protects the server and the other clients, and is the honest move when a client genuinely cannot keep up. Pair it with resume so the disconnect is not data loss.

The decision rule: **if every event must be delivered, you must be able to slow or persist — block a controllable source, or persist events and let the client resume. If only the latest value matters, drop or coalesce and never block.** Mixing these up is how you get either an OOM (kept everything) or a silent gap (dropped something that mattered). Name, per stream, which guarantee you are making, and pick the strategy that fits it.

### What the transport gives you for free, and what it does not

A fair question: doesn't TCP already do flow control? Yes — and this is where the transports differ in how much backpressure you inherit versus must build. TCP has a receive window: a slow receiver advertises a small window, the sender stops sending when the window is full, and the stall propagates back to whoever is writing to the socket. So at the byte level, a slow client *does* eventually slow your `send()` calls.

The trap is what happens *above* the socket. When your application reads events from the source and writes them to the socket, a full TCP window makes the write block (or, on a non-blocking socket, return "would block"). If your code then *buffers* the event in application memory and moves on to the next one — which is the obvious, naive thing to do — you have defeated TCP's backpressure entirely; the kernel's window is full but your heap is now the unbounded buffer. **TCP's backpressure only helps if you let the write-block propagate up to the source** instead of swallowing it into an app-level queue. This is why "block the producer" is the simplest correct strategy when the source is controllable: you let the slow `send()` naturally throttle your read from the source.

**gRPC over HTTP/2** does the most for you here. HTTP/2 has its own per-stream flow-control windows layered on TCP, and gRPC libraries surface it: when a server-streaming call's client is slow, the library stops the server's `yield` from proceeding until the window opens, so a well-behaved gRPC handler that just iterates and yields gets backpressure propagated to the application layer automatically. **WebSocket and SSE give you less:** the transport backpressure is there at the TCP layer, but the common libraries make it easy to fire-and-forget into an app buffer, so you must consciously check "is the socket's write buffer full?" and stop reading the source when it is. The lesson is not "the transport handles it" — it is "the transport handles it *only if your application code does not paper over the block with an unbounded queue.*" Bound every buffer you control, and let the full-buffer signal travel back to the source.

#### Worked example: backpressure on the payouts dashboard versus a price feed

The same server, two streams, two correct answers. The **payout status** stream carries state changes that must not be lost (a missed `paid` is a wrong dashboard). So we make it resumable: persist events with monotonic ids, bound the per-connection buffer, and if a client overflows it, disconnect with close code `1013` (try again later) — the client reconnects, sends its last cursor, and we replay the gap from the persisted log. No event is ever lost; a slow client just gets a brief reconnect. The **live price** stream carries last-value-wins data, so we do the opposite: a one-slot buffer per client holding only the latest price per symbol, overwritten on each update, flushed when the socket drains. A slow client simply skips intermediate prices and always sees the current one. Same backpressure problem, opposite resolution, because the *meaning* of the data differs. That is the whole discipline.

## Heartbeats, keepalive, and idle timeouts

A long-lived connection has a quiet enemy: **the idle timeout.** Load balancers, reverse proxies, and NAT gateways all reap connections that go silent for too long — often 60 seconds, sometimes less — to reclaim resources. If your stream has nothing to send for a couple of minutes (a calm market, a payout sitting in `pending`), an intermediary will close it out from under you, and depending on the layer, neither end may notice promptly. The connection is dead; the client thinks it is alive; events vanish silently.

The fix is a **heartbeat**: deliberate, low-cost traffic that keeps the connection observably alive and lets each side detect a dead peer. The mechanism differs by transport:

- **SSE:** send a comment line, `: heartbeat\n\n`, every 15 to 30 seconds when idle. The client ignores it; the proxy sees traffic and keeps the connection.
- **WebSocket:** use the protocol's built-in **ping/pong** control frames — the server sends a `ping`, the client's library auto-replies `pong`. No `pong` within a timeout means the peer is gone; close and (client side) reconnect. Many setups *also* run an application-level ping/pong to detect a hung-but-connected peer.
- **Long polling:** less of an issue, since each request is short-lived by design, but keep your wait window under the proxy's idle cap (the 25-to-30-second guidance from earlier).
- **gRPC:** HTTP/2 has its own keepalive pings; configure the interval and timeout on both client and server.

Set the heartbeat interval comfortably under the smallest idle timeout in your path. If your load balancer reaps at 60 seconds, a 20-to-30-second heartbeat gives margin. And tune the load balancer's idle timeout *up* for streaming routes if you can — many default to 60 seconds, which is fine with heartbeats but generous-by-default avoids surprises.

## Reconnection and resuming from a cursor

Networks drop connections. Mobile clients switch from Wi-Fi to cellular. Servers deploy and shed their connections. A streaming API is not "real-time" if a reconnection loses data — it is real-time *and* lossy, which on a payments surface is worse than slow. So reconnection-with-resume is not an add-on; it is core to the contract.

The pattern is the same across transports, and we have already seen the pieces:

1. **Every event carries a monotonic, server-assigned id (a cursor).** Sequential ints or opaque ordered tokens both work; the only requirement is that the server can resume from any id it issued.
2. **The client remembers the last id it successfully processed** — not the last it *received*, the last it *processed*, so a crash mid-processing replays rather than skips.
3. **On reconnect, the client sends that id** — `Last-Event-ID` for SSE (automatic), a `resume` message or `resume_from` field for WebSocket and gRPC (you build it).
4. **The server replays from `cursor + 1`**, which requires it to have **persisted or buffered** enough history. This is the part that turns into real engineering: events must live somewhere durable (a log, a stream store, a database table) for at least your maximum expected disconnect window. Cross-link to the message-queue series for the log model that makes this natural — a partitioned append-only log is precisely a "resume from offset" substrate.

The subtle correctness point: **resume must be idempotent on the client.** Because the client tracks the last *processed* id and the network is at-least-once, a reconnect can legitimately redeliver the last event the client already saw. The client must tolerate seeing an id it has handled — dedupe on id, or make the apply operation idempotent. This is the same at-least-once reality the idempotency-keys post in this series wrestles with, surfacing here in stream form: the transport gives you at-least-once; exactly-once is something the *client* constructs by deduping on the cursor.

## Authentication on a long-lived connection

Authenticating a request that lives for milliseconds is easy — put a bearer token in the `Authorization` header and check it. Authenticating a connection that lives for an *hour* is genuinely harder, because the obvious facts of short-lived auth stop holding: the token may expire while the connection is still open, and for some transports you cannot even set the header you want.

![a graph of authenticating and routing a long-lived connection where the token is verified at the handshake then a sticky route pins the session and an expiry timer forces re-auth](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-8.png)

Three distinct problems, three answers:

**1. Authenticating at connection time.** For **SSE** and **WebSocket** the connection *starts* as an HTTP request, so you authenticate it like any HTTP request — a `Bearer <token>` in the `Authorization` header, verified at the handshake before you accept the stream. There is one infamous gap: the browser's `EventSource` and `WebSocket` constructors do **not** let you set custom request headers, so you cannot put `Authorization` there from browser JavaScript. The workarounds, least-bad first: rely on a same-site **cookie** (sent automatically, but mind CSRF and `SameSite`); pass a **short-lived, single-use ticket** as a query parameter that you minted from a normal authenticated request moments earlier (acceptable because it is short-lived and one-use, but it can land in logs, so keep its lifetime to seconds); or, for WebSockets, send the token as the **first message** after `open` and reject the connection if it does not arrive promptly. Avoid putting a long-lived bearer token in the URL query string — URLs leak into access logs, proxies, and browser history.

**2. Token expiry mid-stream.** A connection open for an hour will outlive a 15-minute access token. You have two honest options. The strict one: when the token expires, the server closes the connection with a clear code (`4401` is a common app-level convention for "auth expired"), and the client silently refreshes its token and reconnects, resuming from its cursor — the user sees nothing. The smoother one: the client refreshes its token in the background *before* expiry and sends a `reauth` message with the new token over the existing connection, which the server validates and which resets the connection's expiry — no disconnect at all. The strict option is simpler and leans on your reconnect-and-resume machinery (which you built anyway); the smooth option is nicer but needs an app-level re-auth message in your protocol. Either is fine; *no* expiry handling is not — a connection authenticated an hour ago with a since-revoked token is a security hole.

**3. Authorization that can change.** Authentication is "who are you"; authorization is "what may you do," and on a long connection the answer can change after the connection opens (a permission revoked, a subscription downgraded). For sensitive streams, re-check authorization periodically on the server, not just at handshake — tie it to the same expiry timer that forces re-auth. The connection being old is not proof the caller is still allowed.

## Scaling stateful connections

Here is where streaming collides hardest with the architecture you are used to. A stateless REST service scales trivially: any instance can serve any request, so you put $N$ instances behind a load balancer and you are done. A streaming service is **stateful** — each connection is pinned to one specific server for its whole life — and that breaks several assumptions at once.

**Sticky sessions.** The load balancer must route a given client's connection to a server and keep it there; for WebSockets this is mandatory (the upgraded connection physically *is* a TCP connection to one box). For SSE and long polling, stickiness helps if you keep per-connection state on the server. Use connection-level affinity (the LB pins the TCP connection, which is automatic for WebSockets) or a routing key. The cost: you cannot freely rebalance, and a deploy that restarts a server *disconnects every connection on it* — which is exactly why reconnect-and-resume is non-negotiable, because rolling deploys will drop connections as a matter of routine, and a good client just reconnects to another node and resumes.

**Fan-out behind a pub-sub edge.** The deeper problem: an event is produced *somewhere* (the payment service marks a payout `paid`), but the client interested in it is connected to *some other* edge node. Node B holds the socket; the event originated near node A. How does it get from A to B? You do **not** try to make every node know about every connection. Instead, the edge nodes that hold the sockets all **subscribe** to a shared **pub-sub** bus (Redis pub/sub, a message broker, a partitioned log). The producer publishes the event once to the bus; the bus fans it out to every edge node; each node forwards it to whichever of *its* sockets care. The edge layer becomes a thin, horizontally scalable shell of socket-holders, and the hard fan-out lives in the bus.

![a fan-out graph where a payment event is published once to a pub-sub bus that delivers it to multiple edge nodes, each forwarding to its own connected dashboards](/imgs/blogs/streaming-apis-sse-websockets-and-server-streaming-4.png)

This is the pattern that makes a million-connection streaming system tractable: stateless producers publish to a bus; a fleet of edge nodes each hold a slice of the connections and subscribe to the bus; clients reconnect to *any* node and resume from a cursor backed by a durable log. For the bus itself — delivery guarantees, ordering, the difference between a transient pub/sub and a durable log you can replay — cross-link out to the message-queue series, which owns those internals; in particular the model that lets a reconnecting client resume from an offset is the **log**, contrasted with queue and pub/sub in [queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models). The choice between an ephemeral pub/sub (cheap, no replay) and a durable log (replayable, the resume substrate) is exactly the choice your resumability contract forces.

**Connection budgets.** Each open connection costs a file descriptor, some kernel socket buffers, and a slice of application memory. The C10K problem — ten thousand connections per box — was solved long ago with async, event-driven servers; modern setups handle hundreds of thousands per node. But it is finite, so capacity-plan in connections, not just requests-per-second, and watch file-descriptor limits (`ulimit`), which bite long before CPU does.

**Idle timeouts at the load balancer, again.** Worth repeating in the scaling context: your LB's idle timeout governs every connection through it. Set it generously for streaming routes and *always* run heartbeats, or your nicely-scaled fleet will quietly shed connections every 60 seconds.

## Observing a streaming API

You cannot operate what you cannot see, and streaming endpoints are invisible to the request-counting dashboards that work for REST. A REST service's health shows up as requests-per-second, error rate, and latency percentiles — the RED metrics. A streaming service spends most of its life inside a *single* long request, so "requests per second" tells you almost nothing and "request latency" is meaningless when the request is supposed to last an hour. You have to measure the things that actually describe a stream's health:

- **Connected connections, as a gauge** — the current count of open streams per node and in total. This is your capacity metric; you scale on it the way you scale REST on RPS. A cliff in this number during a deploy is normal (connections shed and reconnect); a cliff at any other time is an incident.
- **Connection lifetime distribution** — how long connections last. A pile-up of very short lifetimes means clients are connecting and immediately dropping, usually a handshake, auth, or proxy problem. A healthy stream has long lifetimes punctuated by reconnects.
- **Events sent per second and per connection**, plus **bytes out** — your actual throughput, and the input to backpressure planning.
- **Per-connection send-buffer depth** — the early-warning signal for the slow-consumer problem. A rising buffer depth on some connections is backpressure building; alert before it becomes an OOM, not after.
- **Reconnect rate and resume-gap size** — how often clients reconnect and how many events they replay on resume. A spike in resume-gap size means clients are missing windows of events, which points at drops or under-provisioned history.
- **Heartbeat misses** — connections where the expected ping/pong or comment did not flow, the signal of a half-open (dead-but-not-closed) connection.

And crucially: **propagate a correlation id onto the connection at the handshake and stamp it on every event and every log line for that connection.** Without it, debugging "this one dashboard stopped updating" across a fleet of edge nodes and a pub-sub bus is nearly impossible — you cannot grep for the connection because it has no request id the way a REST call does. Assign one at `open`, log it, and put it in the events. For the broader observability toolkit — RED metrics, tracing, and SLOs for an API — the operability posts later in this series go deeper; the point here is only that streaming needs a *different* metric set than request/response, and the difference catches teams off guard.

## Case studies: how the big ones actually stream

A few accurate, real-world anchors — because the abstract advice lands harder when you see who made which call.

**LLM token streaming over SSE.** The dominant pattern for chat-model APIs is to stream the generated tokens to the client over **Server-Sent Events**. The client opens a request, the server streams `data:` events each carrying a chunk of the completion, and a terminal sentinel (commonly a `data: [DONE]` line) marks the end. SSE is the right call here precisely because the flow is *one-way* (server to client — the prompt went up in the initial request body, the tokens come down) and *text* (it is generated text), which is exactly SSE's sweet spot, with proxy-friendliness and free reconnection as bonuses. It is a clean illustration of "direction decides the transport": there is no need for a bidirectional WebSocket when the client only receives.

**Chat and trading over WebSockets.** Interactive, bidirectional surfaces — team chat where you both read and post, trading interfaces where you both watch prices and place orders on the same live channel — are the canonical WebSocket use case, because the client genuinely sends over the same connection it receives on. Many real-time messaging and market-data platforms expose a WebSocket where the client sends subscribe and command messages and receives a stream of events back, often with an app-level ping/pong heartbeat and a cursor-or-sequence-number scheme for resuming after a drop — exactly the "you build resume yourself" reality described above, because WebSockets give you no `Last-Event-ID` equivalent.

**SSE for dashboards and progress.** Live dashboards and progress indicators — server pushes state, client displays it — are a natural SSE fit, which is why it is a common choice for status pages, build/CI progress, and operational dashboards like our payouts wall. The browser's `EventSource` plus `Last-Event-ID` gives you reconnection and gap-free resume with almost no client code, and because it is plain HTTP it deploys behind the same gateways and CDNs as the rest of the API.

**gRPC streaming inside the fleet.** For service-to-service streaming — one internal service watching another's event flow, or a data pipeline pushing a stream of records — **gRPC server/bidirectional streaming** over HTTP/2 is the common internal choice, for the typed `.proto` contract, polyglot codegen, deadlines, and transport-level flow control. It rarely faces a browser directly; when a browser needs the data, the stream is terminated at the edge and re-emitted as SSE or WebSocket.

The through-line: the big systems do not pick a streaming transport by fashion. They pick by *direction* and *who the client is* — one-way to a browser leans SSE, two-way to a browser is WebSocket, inside-the-fleet is gRPC streaming — which is exactly the decision tree above.

## When to reach for streaming (and when not to)

A decisive recommendation section, because every choice here is a trade-off and the failure mode is usually over-engineering.

**Reach for SSE when** the flow is one-way (server to client), the client is a browser or HTTP client, and the payload is text or JSON: live prices, progress bars, notifications, LLM tokens, dashboards. It is the most under-used right answer in this space. Serve it over HTTP/2 if you lean on it heavily.

**Reach for WebSockets when** — and ideally only when — the client must send over the same live channel: chat, collaborative editing, multiplayer games, an interactive console, a trading UI that takes orders on the price socket. Accept that you will hand-build reconnection, resume, and heartbeats.

**Reach for gRPC streaming when** the parties are services inside your system (or polyglot clients you control), you want a typed contract and codegen, and you can run HTTP/2 end to end. Terminate it at the edge for browsers.

**Reach for long polling when** your infrastructure breaks the fancier options and you need something that works through anything that speaks HTTP. It is a fallback, not a default.

And do **not** stream when:

- **A snapshot answers the question.** If the client wants the current state once, a `GET` with an `ETag` and conditional requests is simpler, cacheable, and cheaper. Do not hold a connection open to deliver one value.
- **Updates are rare and non-urgent.** A poll every minute beats a per-client held connection for something that changes twice a day. The connection's standing cost is not worth it.
- **You only need server-to-client but reflexively reach for WebSockets.** You will rebuild SSE's free reconnection by hand for no gain. Direction first.
- **You cannot hold connections open reliably** (some serverless platforms cap execution time, some proxies kill long connections). Either fix the infrastructure or fall back to long polling — do not pretend a connection you cannot hold is a stream.
- **You have not designed backpressure and resume.** A streaming endpoint without a bounded buffer and a cursor is a future OOM and a future silent gap. If you are not going to build those, you are not ready to stream; ship polling and come back when you are.

## Key takeaways

- **Direction decides the transport.** One-way to a browser is SSE; two-way to a browser is WebSocket; inside the fleet is gRPC streaming; the universal fallback is long polling. Choose by force, not fashion.
- **SSE is the under-used right answer** for server-push of text and JSON. Plain HTTP, proxy-friendly, and `EventSource` plus `Last-Event-ID` gives you reconnection and gap-free resume for free.
- **WebSockets give you the client-to-server channel but no free reconnection.** You build backoff, resume-from-cursor, and heartbeats yourself. Only pay that cost when you truly need bidirectional.
- **An unbounded per-connection buffer is a memory leak.** Bound it, and decide per stream what to give up when it fills — block a controllable source, drop superseding data, coalesce by rate, or disconnect-and-resume. Match the strategy to whether every event must survive or only the latest value matters.
- **Resumability is a contract, and the cursor makes it provable.** Monotonic event ids plus persisted history plus resume-from-`cursor+1` is the difference between a stream you can trust and one that lies by omission. Make the client dedupe on id — the transport is at-least-once.
- **Heartbeats keep connections alive** under idle-timeout-happy proxies and load balancers. SSE comments, WebSocket ping/pong, HTTP/2 keepalive — set the interval under the smallest idle cap in the path.
- **Authenticate at the handshake and handle expiry.** Browser constructors cannot set headers, so use a cookie or a short-lived single-use ticket; close-and-reconnect or re-auth-in-band when the token expires; re-check authorization on long connections.
- **Scale by making the edge thin and the bus do the fan-out.** Sticky connections, a pub-sub bus (or a durable log for replay) behind the edge, and clients that reconnect to any node and resume — that is how a million connections stays tractable.

## Further reading

- **HTML Living Standard — Server-Sent Events** (the `text/event-stream` format, `EventSource`, and `Last-Event-ID` reconnection): the canonical SSE spec in the WHATWG HTML standard.
- **RFC 6455 — The WebSocket Protocol** (the `Upgrade` handshake, `Sec-WebSocket-Key`/`Accept`, framing, and control frames).
- **gRPC documentation — streaming RPCs** (server, client, and bidirectional streaming over HTTP/2, deadlines, and flow control), and the Protocol Buffers language guide.
- **RFC 9110 — HTTP Semantics** (the methods, status codes, and connection semantics that long polling and the SSE/WebSocket handshakes build on).
- Within this series: the intro hub, [what is an API: the contract between systems](/blog/software-development/api-design/what-is-an-api-the-contract-between-systems); the companion [event-driven and async APIs: webhooks, pub/sub, and AsyncAPI](/blog/software-development/api-design/event-driven-and-async-apis-webhooks-pubsub-and-asyncapi); [gRPC and Protocol Buffers: contracts, codegen, and streaming](/blog/software-development/api-design/grpc-and-protocol-buffers-contracts-codegen-and-streaming); [long-running operations: async jobs, polling, and callbacks](/blog/software-development/api-design/long-running-operations-async-jobs-polling-and-callbacks); and the capstone [the API design playbook](/blog/software-development/api-design/the-api-design-playbook-a-review-checklist-first-endpoint-to-v2).
- For the broker internals behind the fan-out — delivery guarantees, ordering, and the log model that backs resume — see [queue vs pub/sub vs log: three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models).
