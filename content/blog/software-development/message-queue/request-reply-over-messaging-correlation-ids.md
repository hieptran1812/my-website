---
title: "Request-Reply Over Messaging: Correlation IDs and Reply Queues"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Build a synchronous-style request/response on top of inherently asynchronous messaging: a request queue, a reply queue, a correlation id to match the answer to the question, a pending-request map, and the timeout handling that no reply forces you to design — plus when you should just call a direct RPC instead."
tags:
  [
    "message-queue",
    "request-reply",
    "rpc",
    "correlation-id",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "scatter-gather",
    "timeouts",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/request-reply-over-messaging-correlation-ids-1.webp"
---

Messaging is asynchronous by nature. You publish a message, the broker buffers it, and some consumer picks it up later — maybe in two milliseconds, maybe in two minutes if the consumer is backed up. There is no "return value." The whole point of a queue is to *decouple* the sender from the receiver in time, so the sender does not block waiting for the receiver to finish. And yet, over and over, real systems need exactly the thing a queue refuses to give them: a caller that says "compute this for me and give me the answer back," and then waits for that one answer, addressed to that one caller, ignoring every other message flying around the broker. That is request-reply, and bolting it onto a message queue is one of the most common — and most commonly botched — patterns in distributed systems.

The trick that makes it work is small and elegant. You send your request to a **request queue**, you tell the server where to send the answer by stamping a **reply-to** address on the message, and you stamp a unique **correlation id** on the request. The server does its work, publishes the answer to your reply address, and copies your correlation id onto the reply. When the answer comes back through your **reply queue**, you read the correlation id, look it up in a table of requests you are still waiting on, find the future or callback that belongs to it, and hand it the answer. That is the entire mechanism. Everything hard about request-reply over messaging is not the happy path — it is the absent reply, the duplicate reply, the reply that arrives after you gave up, and the question of whether you should have done any of this instead of just calling the service directly.

![A directed graph showing a client publishing a request carrying a correlation id into a request queue, a server consuming and processing it, then publishing a reply with the same correlation id into a reply queue that the client consumes and matches](/imgs/blogs/request-reply-over-messaging-correlation-ids-1.webp)

By the end of this post you will be able to implement request-reply on RabbitMQ correctly, including RabbitMQ's built-in direct reply-to optimization that saves you from creating a queue per call. You will know the two reply-queue strategies — one shared queue filtered by correlation id versus a temporary exclusive queue per client — and exactly which costs each one trades away. You will be able to build the pending-request map that ties a correlation id to a waiting future and a deadline, and the timeout sweep that cleans up the requests whose replies never came. You will understand why this pattern is natural on a routing broker like RabbitMQ and genuinely awkward on a log like Kafka. You will be able to extend it to scatter-gather, where one request fans out to many workers and their replies are aggregated. And — most importantly — you will have a clear, defensible answer to the question that should precede all of this: *should I be doing request-reply over a message queue at all, or should I just make a direct call?* If you have read the companion posts on [the three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) and [push versus pull](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read), this post is where those foundations get bent into a shape they were never designed for, and where you learn exactly how much that bending costs.

## 1. Why you would do RPC over a message queue at all

Start with the honest version of the question, because it is the one that matters most and the one most architecture documents skip. A remote procedure call is a solved problem. You have gRPC, you have plain HTTP, you have Thrift, you have a dozen RPC frameworks that give you a typed `result = service.compute(request)` with a connection, a timeout, retries, and load balancing baked in. They are fast, they are simple to call, and a junior engineer can use one correctly on day one. So why would any sane person take that clean synchronous call and route it through a broker, adding two extra network hops, a correlation id, a reply queue, and a pending-request map?

The answer is never "for speed." Request-reply over a message queue is *always slower* than a direct RPC for the same logical call, because you have inserted a broker between the two parties and the message now has to be enqueued, persisted (maybe), dequeued, processed, enqueued again on the reply path, persisted again, and dequeued again. You pay for that. If your only goal is to get an answer back as fast as possible, do not do this. Make the direct call.

You do request-reply over a queue for three reasons, and you should be able to name which one applies before you write a line of code.

The first is **decoupling and location transparency**. With a direct RPC, the caller must know how to reach the server: a hostname, a port, a service-discovery lookup, a load balancer in front. The caller is bound to the topology. With request-reply over a broker, the caller knows only the name of a queue. It has no idea how many servers are consuming that queue, where they live, whether they were just redeployed, or whether they are behind three layers of NAT. You can scale the server fleet from one to fifty, move it to another data center, or restart all of it, and the caller's code does not change and the caller's connections do not break. The broker is the rendezvous point, and the queue name is the only contract.

The second is **load leveling**. A direct RPC has no buffer. If a thousand callers hit your service at once and the service can handle four hundred concurrent requests, six hundred of them get connection refused, time out, or pile up in a thread pool until it melts. A request queue is a buffer. Those thousand requests sit in the queue, and the server fleet drains them at whatever rate it can sustain. The queue absorbs the spike. Callers still wait — possibly a long time, which is its own problem you must handle with timeouts — but the server is never overwhelmed, and you never get the cascading failure where an overloaded service drags down everything that calls it. This is the same load-leveling argument that motivates queues in general, applied to a request/response workload.

The third is **resilience across deploys and partial failures**. With a direct RPC, if the server is down when you call it, your call fails immediately. With request-reply over a durable queue, if the server fleet is briefly down — rolling deploy, a crash, a five-second blip — the request waits in the queue and gets processed when a server comes back, as long as that happens within your timeout. The broker decouples the *availability* of the caller from the availability of the server in a narrow window. You do not get this for free — a request that waits too long must time out — but for workloads where a few seconds of delay is acceptable and a hard failure is not, the buffer is genuinely valuable.

![A linear pipeline of five stages: send request with correlation id and reply-to, enqueue at the broker, server processes the work, publish reply with the same correlation id, and finally match the correlation id to resolve the waiting future](/imgs/blogs/request-reply-over-messaging-correlation-ids-2.webp)

Notice what is *not* on that list: low latency, simplicity, or ease of debugging. Those all get worse. The pipeline in the figure above shows why: every request you make now crosses the broker twice, once on the way out and once on the way back, and each crossing is an enqueue and a dequeue with its own latency and its own opportunity to fail. If you cannot point at decoupling, load leveling, or cross-deploy resilience as the reason you are doing this, you are paying for a return trip you did not need. Make the direct call and move on.

#### Worked example: the latency budget, direct RPC versus a broker round trip

Put numbers on the cost so the trade is not abstract. A direct gRPC call to a service on the same network has a realistic p99 of around 5 milliseconds end to end: serialize the request, one network round trip, the server processes, serialize the response, one network round trip back. That 5 milliseconds is dominated by the actual work plus two small network legs. Now route the same logical call through a broker. The request must travel from the client to the broker (one network leg), be enqueued and possibly persisted, be dequeued and delivered to the server (a second network leg), get processed, then the reply travels from the server to the broker (a third leg), gets enqueued, then is delivered back to the client (a fourth leg). You have gone from two network legs to four, plus two enqueue/dequeue cycles in the broker, plus any persistence the broker does. On a healthy lightly-loaded broker, each enqueue-plus-dequeue is on the order of 1 to 3 milliseconds, and you have two of them, so even with the server processing taking the same time, the broker round trip adds roughly 4 to 8 milliseconds of pure transport overhead. A call that was 5 milliseconds direct becomes 10 to 15 milliseconds over the broker on a *good* day. That is a 2x to 3x latency multiplier for the privilege of decoupling, and it gets dramatically worse the instant the broker is under load and requests queue. If you are spending a 5-millisecond direct call on a 12-millisecond broker round trip, the 7 extra milliseconds had better be buying you decoupling or load leveling you genuinely need — because as raw latency, it is a pure loss.

### The synchronous illusion

The thing to keep in your head is that request-reply over a queue is an *illusion of synchrony* layered on top of fundamentally asynchronous transport. The caller's code looks synchronous — `result = await client.call(request, timeout=2.0)` — but underneath, the call is implemented as "publish a message, then wait for an unrelated message to arrive and match it back to me." The broker has no notion that these two messages are related. It does not know the reply belongs to the request. *You* know, because you stamped a correlation id on both and you kept a table. The synchrony lives entirely in your client library, not in the broker. The broker is just moving independent messages around, exactly as it always does. That mental model — synchrony is faked by the client, the broker stays dumb — is the key to understanding every design decision that follows.

## 2. The request-reply pattern and the correlation id

Let us build the pattern precisely, piece by piece, because every piece exists to solve a specific problem and skipping any one of them breaks the whole thing.

The first piece is the **request queue**. This is an ordinary queue. The client publishes request messages into it, and one or more server instances consume from it as competing consumers — each request is processed by exactly one server, the same way a [work queue](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) distributes tasks. Nothing special here. If you have ten server instances, requests are load-balanced across them by the broker.

The second piece is the **reply-to address**. When the server finishes processing a request, it needs to know where to send the answer. The server cannot guess — it has no idea which client sent the request, and even if it did, the client is not addressable as a server. So the client tells the server explicitly, by putting a reply address in the message. In AMQP this is the standardized `reply_to` property: the client sets it to the name of a queue it is consuming from, and the server reads that property and publishes the reply to the default exchange with the routing key equal to that queue name. The reply-to is part of the message envelope, a first-class property, not something you have to invent in the body. This is what gives you location transparency on the reply path: the server does not know who the client is, only the name of a queue to drop the answer into.

The third piece — the heart of the whole pattern — is the **correlation id**. Here is the problem it solves. Suppose your client has sent three requests, A, B, and C, all from the same process, all using the same reply queue, and all roughly at the same time. Three replies are going to come back through that one reply queue. Which reply belongs to which request? The replies do not necessarily come back in order — server A might be slow and server C fast, so C's reply arrives first. If you naively assumed replies match requests in send order, you would hand C's answer to the code waiting for A's answer, and now your system is silently corrupting data. The correlation id fixes this. The client generates a unique id for each request — a UUID is the standard choice — and stamps it on the request as the `correlation_id` property. The server, when it builds the reply, *copies that exact correlation id* onto the reply message. The server does not generate a new one; it echoes the client's. Now when a reply comes back, the client reads its correlation id, looks up which request had that id, and routes the answer to the right waiting caller. The correlation id is the thread that ties a specific reply to a specific request across an asynchronous, unordered, multiplexed channel.

```python
import uuid
import pika  # RabbitMQ client

# CLIENT SIDE: build and publish a request
corr_id = str(uuid.uuid4())              # unique per request
channel.basic_publish(
    exchange="",
    routing_key="rpc_request_queue",     # the request queue
    properties=pika.BasicProperties(
        reply_to="rpc_reply_queue_for_this_client",  # where to send the answer
        correlation_id=corr_id,          # the thread tying request to reply
    ),
    body=b'{"op":"factorial","n":12}',
)
```

On the server side, the logic is symmetric and dumb on purpose. The server consumes a request, does the work, and publishes a reply whose routing key is the request's `reply_to` and whose `correlation_id` is copied straight from the request:

```python
# SERVER SIDE: process a request and reply
def on_request(ch, method, props, body):
    result = compute(body)                # do the actual work
    ch.basic_publish(
        exchange="",
        routing_key=props.reply_to,        # echo the client's reply address
        properties=pika.BasicProperties(
            correlation_id=props.correlation_id,  # echo the client's corr-id
        ),
        body=result,
    )
    ch.basic_ack(delivery_tag=method.delivery_tag)
```

That is the entire server. It never looks at the correlation id's value — it just copies it. It never decides where to reply — it just reads `reply_to`. The server is stateless with respect to the request-reply machinery; all the state lives on the client. That asymmetry is deliberate and it is what makes the pattern scale: you can run a hundred identical, stateless server instances, and the correlation-and-matching complexity is entirely the caller's problem.

### Why a UUID and not a counter

People sometimes ask why the correlation id is a UUID rather than a simple incrementing integer, which would be cheaper. Within a single client process, a counter would actually be fine — you control it, it is unique within your process, and your pending map is local. The reason to prefer a UUID is that correlation ids leak across process boundaries the moment you have more than one client, share a reply queue, restart a client, or replay a message. Two clients both starting their counters at 1 will collide. A client that restarts and resets its counter to 1 will reuse ids that an in-flight reply from before the restart might still match. A UUID sidesteps all of that: it is globally unique with no coordination, so a correlation id minted anywhere by anyone never collides with one minted anywhere else. The cost is sixteen bytes per request, which against the size of a real request payload and the cost of a broker round trip is nothing. Use a UUID. The one time a counter bites you it will be a heisenbug that corrupts a customer's data, and you will not enjoy debugging it.

### Out-of-order replies, made concrete

It is worth tracing exactly why the correlation id is non-negotiable, because the failure it prevents is subtle and only appears under concurrency. Picture a single client that issues three requests in quick succession: request A asks for a heavy computation, request B asks for a light lookup, request C asks for a medium one. The client sends them in the order A, B, C, all into the same request queue, all with the same reply queue. The broker distributes them across three server instances. Server A is slow because A is heavy; server B is fast because B is light. So the replies come back in the order B, then C, then A — *the reverse-ish of the send order*. Now imagine the client tried to match replies by *send order*, assuming the first reply belongs to the first request. The first reply to arrive is B's, but the client hands it to the code waiting for A. A's caller now has B's answer. C's reply arrives and goes to B's caller. A's reply finally arrives and goes to C's caller. Every single answer is delivered to the wrong waiter, and nothing crashes — the data is just silently, catastrophically wrong. The correlation id makes this impossible: each reply carries the id of the request it answers, so B's reply matches B's pending entry no matter when it arrives or what order it arrives in. Order independence is the entire point. The reply queue is an unordered, multiplexed channel by design, and the correlation id is what lets you demultiplex it correctly.

This is also why you cannot lean on the broker to "keep replies in order." Even if the broker preserved order within the reply queue, the *servers* produce replies out of order because they finish at different times, and the order in which replies are published is the order they finished, not the order the requests were sent. There is no level of the stack at which send-order equals reply-order, so there is no shortcut around carrying an explicit correlation id. Anyone who tells you they matched replies by order got lucky in testing because their requests all took the same time, and they will get a production incident the day one request takes longer than another.

## 3. Reply-queue strategies: exclusive vs shared

Now the first genuinely interesting design decision. The client needs a reply queue — somewhere for answers to land. There are two fundamentally different ways to provision it, and the choice has real operational consequences.

![A two-panel comparison: a shared reply queue used by all clients that must filter every reply by correlation id but has cheap setup, versus an exclusive reply queue created per client that needs no filtering but causes connection and queue churn](/imgs/blogs/request-reply-over-messaging-correlation-ids-3.webp)

The first strategy is a **temporary exclusive reply queue per client**. When a client process starts up (or, in the most extreme version, for every single request), it declares a brand-new queue — auto-delete, exclusive to its own connection, with a server-generated random name. It consumes only from that queue. It puts that queue's name in the `reply_to` of every request. Because the queue is exclusive and used by exactly one client, *every reply that arrives in it belongs to this client*. There is no filtering to do — if a message landed in my private queue, it is for me. The matching step is still needed (the client has multiple in-flight requests and must match each reply to the right one by correlation id), but there is never any question of a reply being meant for *someone else*. The reply queue is private property.

The cost of the exclusive strategy is **queue and connection churn**. Creating a queue is not free on the broker — it is a metadata operation, it has to be declared, tracked, and torn down. If you create one exclusive queue per *client process* and reuse it for the life of the process, this is cheap and totally fine; most well-written RabbitMQ RPC clients do exactly this. But if you naively create one exclusive queue per *request* — which a surprising number of tutorials and copy-pasted implementations do — you are now creating and destroying a queue for every single call. At a few hundred requests per second, that is a few hundred queue churns per second, which shows up as elevated broker CPU, management-plane pressure, and in pathological cases, broker instability. The exclusive-queue-per-request anti-pattern is one of the most common ways people accidentally DoS their own RabbitMQ.

The second strategy is a **single shared reply queue** used by many clients (or many requests from one client). All clients consume from one well-known reply queue. Every reply lands in that one queue. The catch: a reply in the shared queue might belong to *any* of the clients consuming it. With competing consumers on a shared queue, a reply meant for client A might get delivered to client B, who has no pending request with that correlation id. Client B has to recognize "this is not mine" and handle it — either by rejecting and requeuing the message (hoping the right client gets it next) or, more robustly, by not using competing consumers at all but instead a routing scheme where each reply is routed to the correct client. The pure shared-queue-with-competing-consumers approach is fragile precisely because the broker has no idea which consumer a reply is for; it just round-robins, and a reply can land on the wrong consumer.

The honest version of "shared reply queue" that actually works is usually **one shared reply queue per client process**, not one shared across all clients. Each client process has its own long-lived reply queue (which it may have created at startup), and within that process it shares that single queue across all the many concurrent requests the process makes. This gives you the best of both: no per-request queue churn (the queue is created once at startup), and no cross-client confusion (the queue is private to one process, so every reply in it is mine — I just have to match it to the right pending request by correlation id). The filtering cost is the correlation-id lookup, which is a hash-map lookup, which is nanoseconds. This is the strategy most production RPC-over-RabbitMQ clients actually use, and it is what RabbitMQ's own tutorial recommends.

The distinction that confuses people is "shared across requests" versus "shared across clients." Sharing a reply queue *across requests within one process* is good and normal — it is how you avoid per-request churn. Sharing a reply queue *across multiple client processes* with competing consumers is the fragile thing. Keep those separate in your head. The table makes the tradeoffs concrete:

| Strategy | Filtering | Setup cost | Failure mode |
| --- | --- | --- | --- |
| Exclusive queue per request | None (queue is private) | High (queue per call) | Broker churn / instability under load |
| Exclusive queue per process | None (queue is private) | Low (queue at startup) | Lost replies if process dies mid-call |
| Shared queue per process | Corr-id lookup (cheap) | Low (queue at startup) | Same — replies in flight lost on crash |
| Shared queue across processes | Corr-id lookup + reject-if-not-mine | Lowest | Cross-client misdelivery, requeue storms |

The pragmatic recommendation: **one reply queue per client process, created at startup, shared across all of that process's concurrent requests, matched by correlation id.** That is the sweet spot. We will see in the next section that RabbitMQ has a built-in feature, direct reply-to, that gives you the benefits of a per-process reply queue without you having to manage a queue at all.

#### Worked example: queue churn under the per-request anti-pattern

Suppose you run an RPC client doing 800 requests per second, and someone wrote it to create an exclusive reply queue per request — declare, consume, publish request, wait, get reply, cancel consumer, delete queue. Each request now triggers a queue declare and a queue delete on the broker: 800 declares and 800 deletes per second, 1,600 management operations per second on top of the actual message traffic. On a healthy single RabbitMQ node, queue declare/delete is on the order of tens of microseconds of broker CPU each in the easy case, but these operations touch the broker's metadata store and serialize through the queue-management machinery, so under concurrency they contend. At 1,600 per second you will typically see broker CPU climb several percentage points purely from churn, mnesia (or the metadata store) under write pressure, and management-UI lag. Now contrast the per-process strategy: that same client creates *one* reply queue at startup. That is 1 declare for the entire lifetime of the process, instead of 800 per second. The message traffic is identical — still 800 requests and 800 replies per second — but the metadata churn drops from 1,600 ops/sec to effectively zero. The lesson: the reply queue's *lifecycle* matters far more than its *throughput*. Tie the queue to the process, never to the request.

## 4. The pending-request map and matching replies

The state that makes the synchronous illusion work lives in one data structure on the client: the **pending-request map**. It is a hash map keyed by correlation id, and each entry holds everything you need to deliver a reply (or a failure) to whoever is waiting.

![A layered stack showing the pending-request map: a correlation id UUID key, the future or promise to resolve, the callback to invoke on reply or error, the deadline timestamp with a timer, and a sweeper that evicts expired entries](/imgs/blogs/request-reply-over-messaging-correlation-ids-5.webp)

Walk through the lifecycle of one entry. When the client makes a call, it generates a correlation id, creates a future (or a promise, or a channel, or a callback registration — the concurrency primitive depends on your language), and inserts `corr_id -> {future, deadline}` into the map *before* publishing the request. The "before publishing" ordering matters: if you published first and the server was blazing fast, the reply could arrive before you finished inserting into the map, and you would drop your own reply because there was nothing to match it to. Always register, then publish.

Then the client publishes the request and awaits the future. The future is unresolved; the calling coroutine or thread is parked. Meanwhile, a separate consumer loop — running on the reply queue, on its own thread or in the client's event loop — is pulling replies as they arrive. For each reply, it reads the correlation id, looks it up in the pending map, and if it finds an entry, it resolves that future with the reply body and removes the entry from the map. The parked caller wakes up with its answer. That is the whole match. Here is the shape of it:

```python
import asyncio, uuid

class RpcClient:
    def __init__(self, channel, reply_queue):
        self.channel = channel
        self.reply_queue = reply_queue
        self.pending = {}                  # corr_id -> asyncio.Future
        # consume replies on a background task
        self.channel.basic_consume(reply_queue, self._on_reply, auto_ack=True)

    def _on_reply(self, ch, method, props, body):
        future = self.pending.pop(props.correlation_id, None)
        if future is None:
            # reply for a corr-id we are not waiting on:
            # late reply after timeout, or a duplicate. Drop it.
            return
        if not future.done():
            future.set_result(body)

    async def call(self, body, timeout=2.0):
        corr_id = str(uuid.uuid4())
        future = asyncio.get_event_loop().create_future()
        self.pending[corr_id] = future     # register BEFORE publish
        self.channel.basic_publish(
            exchange="", routing_key="rpc_request_queue",
            properties=pika.BasicProperties(
                reply_to=self.reply_queue, correlation_id=corr_id),
            body=body,
        )
        try:
            return await asyncio.wait_for(future, timeout)
        finally:
            # clean up the entry no matter how this ends
            self.pending.pop(corr_id, None)
```

Several subtle things are happening here that separate a toy from a production client.

First, `self.pending.pop(props.correlation_id, None)` in `_on_reply` returns `None` when the reply does not match any pending request. That is not an error condition you can ignore — it is a normal, expected case. A reply with no matching pending entry means one of three things: the request already timed out and we removed its entry (so this is a *late reply*), the broker redelivered a reply we already processed (a *duplicate*), or — if you got your queues confused — a reply for a different client. In all three cases, the right behavior is usually to drop it silently, because there is no longer anyone waiting for it. But you should *count* these events on a metric, because a sudden spike of unmatched replies tells you your timeouts are too aggressive or your server got slow.

Second, the `finally` block pops the entry on every exit path — success, timeout, or exception. If you only removed entries on successful reply, then every timed-out request would leave a dead entry in the map forever, and your pending map would grow without bound until you ran out of memory. The pending map is a place memory leaks love to live. Clean up on every path.

Third, the `if not future.done()` guard before `set_result` prevents a crash when a reply arrives for a future that the timeout already failed. There is an inherent race: the timeout fires and rejects the future at almost the same instant the reply arrives and tries to resolve it. Without the guard, you call `set_result` on an already-completed future and get an exception in your consumer loop, which can wedge the whole reply consumer. Guard it.

### The matching invariant

The one invariant you must preserve is: **a correlation id is unique among all currently-pending requests on this client.** It does not have to be globally unique forever (though a UUID makes it so for free); it only has to be unique among the requests you are *currently waiting on*. If you reused a correlation id for a new request while an old request with the same id was still pending, a reply could match the wrong future. UUIDs guarantee this trivially. If you use a counter, you must guarantee you never reuse a value while its predecessor is still in the map — which in practice means a counter large enough that wraparound takes longer than your maximum timeout, which is more bookkeeping than just using a UUID.

## 5. Timeouts: designing for the missing reply

Here is the single most important sentence in this entire post: **in request-reply over a message queue, no reply is the common case you must design for, not an exception you handle as an afterthought.** A direct RPC fails loudly — the connection resets, you get an error immediately. Request-reply over a queue fails *silently*: you publish a request, and then nothing comes back, ever, and your future just sits there parked forever unless you armed a timeout. The server might have crashed mid-request. The reply might have been published to a reply queue that no longer exists because your client restarted. The request might still be sitting in the request queue behind ten thousand others. The broker might have dropped the reply. None of these produce an error signal at the client. The only thing the client observes is *silence*. If you do not have a timeout, silence is forever.

![A timeline of one request racing its timeout: at T plus zero the request is sent and a two-second timeout is armed, the broker enqueues it, and then either a reply arrives at forty-five milliseconds and the correlation id matches, or the two-second deadline fires with no reply and the future is rejected with a timeout error](/imgs/blogs/request-reply-over-messaging-correlation-ids-4.webp)

The timeline in the figure shows the race that every request runs. The moment you send a request, you arm a deadline. From then on it is a race between two events: the reply arriving and matching, or the deadline firing. Exactly one of them wins. If the reply wins, you resolve the future and cancel the timer. If the deadline wins, you reject the future with a timeout error and — crucially — you remove the pending entry so a late reply has nothing to match. Designing the timeout *is* designing the pattern. The happy path is trivial; the timeout path is where all the engineering is.

### Choosing the timeout value

The timeout has to be longer than the realistic worst-case round trip, including queueing delay, or you will time out requests that were going to succeed. The realistic worst case is *not* the server's processing time. It is the server's processing time plus the time the request spent waiting in the request queue plus both enqueue/dequeue latencies plus the reply's queueing time. Under load, the queueing delay dominates everything else. It helps to lay the budget out as a table so you can see where the time actually goes:

| Component of the round trip | Light load | Heavy load (backlog) |
| --- | --- | --- |
| Client to broker (enqueue request) | ~1 ms | ~1 ms |
| Wait in request queue | ~0 ms | seconds (depth / drain rate) |
| Server processing | 40 ms | 40 ms |
| Server to broker (enqueue reply) | ~1 ms | ~1 ms |
| Wait in reply queue | ~0 ms | tens of ms |
| Broker to client (deliver reply) | ~1 ms | ~1 ms |
| **End-to-end total** | **~43 ms** | **seconds** |

The table makes the trap obvious: under light load the round trip is dominated by the 40 milliseconds of real work and a timeout of a few hundred milliseconds is plenty. But under a backlog, the "wait in request queue" row explodes from zero to seconds, and a few-hundred-millisecond timeout would fail every request even though the system is healthy and would have answered. The timeout must be sized for the *backlog* case, not the *light* case, which means it must be generous enough to ride out a transient backlog without giving up on requests that are merely waiting their turn. If your server takes 40 milliseconds to process a request but there are 5,000 requests queued ahead of yours and the fleet drains 1,000 per second, your request will not even *start* processing for 5 seconds. A 2-second timeout would fail every request during that backlog even though the system is working correctly and would have answered in 5 seconds. So the timeout must account for queue depth, which means it is workload-dependent and you should set it from observed end-to-end latency percentiles, not from the server's isolated processing time. A common, defensible approach: set the timeout to a few times the observed p99 end-to-end latency, and alarm when timeouts exceed a small percentage of requests, because a rising timeout rate is your earliest signal of a backlog.

### The timeout sweep

There are two ways to implement timeouts, and the difference matters at scale. The simple way is per-request timers: every call arms its own timer (in the code above, `asyncio.wait_for` does this). This is clean and precise but creates one timer per in-flight request, and at high concurrency — tens of thousands of in-flight requests — that is a lot of timer objects and a lot of timer-wheel churn. The scalable way is a **timeout sweep**: instead of a timer per request, you store a deadline timestamp on each pending entry, and a single background sweeper runs periodically (say every 100 milliseconds), scans for entries whose deadline has passed, and rejects them in a batch. This trades timeout precision (you might fire a timeout up to one sweep interval late) for a constant, small number of timers regardless of how many requests are in flight. For a client doing tens of thousands of concurrent requests, the sweep is the right design.

```python
import time

def sweep_expired(self):
    now = time.monotonic()
    expired = [cid for cid, e in self.pending.items() if e["deadline"] <= now]
    for cid in expired:
        entry = self.pending.pop(cid, None)
        if entry and not entry["future"].done():
            entry["future"].set_exception(TimeoutError(f"no reply for {cid}"))
    # schedule the next sweep
    self.loop.call_later(0.1, self.sweep_expired)
```

#### Worked example: timeout sweep cost at 10k in-flight requests

Take a busy client with 10,000 requests in flight at any moment — a fan-out gateway calling a slow backend over a queue. With per-request timers, you have 10,000 live timer objects; every send arms one and every reply or timeout cancels one, so at 10,000 requests per second of throughput you are arming and canceling 20,000 timers per second, and the timer wheel is doing real work. With a sweep instead, you have exactly *one* timer — the sweeper — firing 10 times per second. Each sweep scans the 10,000-entry map: that is a linear pass over 10,000 entries, which at, say, 20 nanoseconds per entry to check a deadline is 200 microseconds of work, done 10 times a second, for 2 milliseconds per second of CPU total — completely negligible. The correlation-id lookup on each *reply* is a single hash-map get, order of 50 to 100 nanoseconds, so 10,000 replies per second costs under a millisecond per second. The whole timeout-and-matching machinery for 10,000 concurrent requests costs single-digit milliseconds of CPU per second. The expensive part was never the matching; it was the per-request timer object churn, and the sweep eliminates it. If you only remember one number from this post: at 10k in-flight, a correlation-id match is a sub-100-nanosecond hash lookup, and a periodic sweep is cheaper than per-request timers by orders of magnitude.

### Retries and the duplicate-reply problem

Once you have timeouts, the natural instinct is to retry: a request timed out, send it again. Be careful. A timeout does not mean the request was *not* processed — it means you did not *hear back in time*. The server may have processed the request and replied; the reply may simply have been slow or lost. If you retry, the server may process the request twice. If the operation is not idempotent, you have now done it twice. This is the same at-least-once duplication problem that haunts all messaging, and it requires the same fix: make the operation idempotent, or deduplicate by a request id on the server. Request-reply does not get a pass on idempotency. A retried request is a duplicate request, and the second reply — arriving after you already gave up on or retried the first — is exactly the unmatched, late reply your `_on_reply` has to drop gracefully. Design for it.

## 6. RabbitMQ direct reply-to in practice

RabbitMQ ships a feature built specifically to make request-reply efficient without forcing you to manage a reply queue at all: **direct reply-to**. It solves the per-process reply-queue problem so cleanly that you should use it as the default for RPC on RabbitMQ unless you have a specific reason not to.

The mechanism is a pseudo-queue with the reserved name `amq.rabbitmq.reply-to`. The client does not declare it — it consumes from it with no-ack (automatic acknowledgement), and then sets the `reply_to` property of its requests to `amq.rabbitmq.reply-to`. RabbitMQ recognizes this special name and routes replies addressed to it directly back to the consuming client over the same channel, with no real queue created on the broker at all. There is no metadata, no queue lifecycle, no churn — the broker holds a lightweight in-memory mapping for the lifetime of the consumer. You get the benefits of a private per-client reply queue (every reply comes straight back to you, no cross-client confusion) with none of the queue-management cost. Here is the client:

```python
# RabbitMQ direct reply-to: no reply queue to declare or delete
class DirectReplyToClient:
    def __init__(self, channel):
        self.channel = channel
        self.pending = {}
        # consume the special pseudo-queue with no-ack (REQUIRED)
        self.channel.basic_consume(
            queue="amq.rabbitmq.reply-to",
            on_message_callback=self._on_reply,
            auto_ack=True,                 # direct reply-to MUST be no-ack
        )

    def _on_reply(self, ch, method, props, body):
        fut = self.pending.pop(props.correlation_id, None)
        if fut and not fut.done():
            fut.set_result(body)

    def call(self, body):
        corr_id = str(uuid.uuid4())
        fut = Future()
        self.pending[corr_id] = fut
        self.channel.basic_publish(
            exchange="", routing_key="rpc_request_queue",
            properties=pika.BasicProperties(
                reply_to="amq.rabbitmq.reply-to",   # the magic name
                correlation_id=corr_id),
            body=body,
        )
        return fut
```

The server code is *unchanged* — it still reads `props.reply_to` (which now contains the special name) and publishes the reply with that as the routing key and the correlation id copied over. The server does not know or care that direct reply-to is in play; it just replies to whatever address the client gave it. That is the beauty of building on the standard `reply_to` property: the optimization is entirely client-side.

There are constraints you must respect. Direct reply-to **requires no-ack** consumption of the pseudo-queue (you cannot ack replies, because there is no real queue to ack against), which means if your client crashes between receiving a reply and acting on it, that reply is gone — you cannot recover it. The reply must be consumed on the **same channel** that is consuming the pseudo-queue. And direct reply-to is **non-durable by design**: it is for genuinely synchronous, request-in-flight RPC where if the client goes away, the answer is moot anyway. If you need the reply to survive a client crash and be picked up later, direct reply-to is the wrong tool — you want a real durable reply queue. But for the overwhelmingly common case of "I am waiting right now for this answer and if I die the answer is useless," direct reply-to is the correct, efficient default. It is the reason you rarely need to hand-roll a reply queue on RabbitMQ.

![A taxonomy tree of reply-queue strategies branching into a shared queue filtered by correlation id with app-side filtering and per-id binding variants, and an exclusive per-client branch with temporary auto-delete queues and RabbitMQ direct reply-to as a broker-managed pseudo-queue](/imgs/blogs/request-reply-over-messaging-correlation-ids-8.webp)

The taxonomy in the figure puts direct reply-to in its place: it is a broker-managed variant of the exclusive-per-client strategy. You get exclusivity (replies come only to you) without managing a queue, because the broker manages the routing for you. On the shared side you have app-side filtering and per-id bindings; on the exclusive side you have temporary auto-delete queues and the direct reply-to pseudo-queue. When someone asks "what reply-queue strategy should I use on RabbitMQ," the answer is almost always the bottom-right leaf: direct reply-to. Reach for a real durable reply queue only when you need a reply to outlive the client.

### A note on the AMQP properties

Three AMQP message properties carry the request-reply pattern, and they are standardized, not RabbitMQ-specific: `reply_to` (where to send the answer), `correlation_id` (the thread tying reply to request), and optionally `message_id` (a unique id for the message itself, useful for server-side dedup). Because these are standard envelope properties, request-reply over AMQP is *interoperable* — a client and server written in different languages, using different AMQP libraries, agree on the contract because the properties are part of the protocol. You are not inventing a convention; you are using one the protocol designers built in for exactly this. That is part of why request-reply feels native on RabbitMQ and foreign on Kafka, which we turn to next.

### The correlation id as a tracing handle

There is a bonus you get nearly for free once you are stamping correlation ids on every request and reply: they double as a distributed-tracing handle. If you log the correlation id at every step — when the client publishes, when the server receives, when the server replies, when the client matches — then a single correlation id becomes a search key that pulls up the entire life of one request across client logs, broker metrics, and server logs. When a request times out and you need to know whether it was lost in the request queue, lost on the server, or lost on the reply path, the correlation id is the string you grep for across all three log streams. For this reason, many teams set the correlation id to (or derive it from) their existing trace id, so the request-reply correlation and the distributed trace share one identifier. That is a small discipline with an outsized payoff during incidents: the difference between "the request vanished somewhere" and "here is exactly where it stopped" is whether you propagated the correlation id into your logs. Treat the correlation id not just as a matching key but as the spine of your request's observability, and the otherwise opaque silence of a missing reply becomes a traceable, debuggable event.

One caveat: do not *reuse* a trace id as a correlation id if a single trace can spawn multiple concurrent requests over the same reply queue, because then two in-flight requests share an id and the matching breaks exactly as the out-of-order scenario warned. The safe construction is correlation id equals trace id plus a per-request suffix, or simply a fresh UUID per request that you *log alongside* the trace id. You want the traceability without sacrificing the uniqueness invariant the pending map depends on.

## 7. Why request-reply is awkward on Kafka

Everything above assumes a broker with the shape of RabbitMQ: named queues you can create cheaply, standard reply-to and correlation-id properties, and consumers that get messages routed to them. Kafka does not have that shape, and trying to do request-reply on Kafka teaches you a lot about why Kafka is a *log*, not a *queue*. If you have read the post on [the three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models), this is the clearest case where the log model fights the pattern.

Start with the reply destination. On RabbitMQ a client creates a private reply queue or uses direct reply-to, and replies come straight back to it. On Kafka there are no per-client queues. There are topics, and topics are partitioned, and consumers in a group are assigned partitions, and the broker has no notion of "route this reply back to the specific client that sent the request." If a client wants replies, it has to consume from some reply *topic*, and that reply topic is shared, partitioned, and read by a consumer group. A reply written to the reply topic does not come back to the specific producer that sent the request; it lands in a partition and is read by whichever group member owns that partition. So unless you are very careful, the reply to client A's request can be read by client B.

People work around this in a few ways, each with sharp edges. One approach is a **reply topic per client instance**, with the client creating a topic at startup and putting its name in the request. But Kafka topic creation is a heavyweight, cluster-wide operation involving the controller and partition assignment — far more expensive than a RabbitMQ queue declare — and you do not want to create topics dynamically per client at any real scale; you will overwhelm the controller and blow up your partition count, which is a finite cluster-wide resource. A second approach is a **shared reply topic partitioned by a client key**, where the client computes which partition its replies will land in and consumes only that partition, and the server writes the reply to that partition by using the right key. This works but it is fiddly: you are now reasoning about partition assignment and key-to-partition hashing to route replies, and it breaks if partitions are reassigned during a rebalance. A third approach is to give up on per-client routing and have *every* client read the *entire* reply topic, each filtering for the correlation ids it cares about and ignoring the rest — which means every client reads every reply, wasting bandwidth proportional to the fan-out, and does not scale past a handful of clients.

There is a deeper mismatch than routing. Kafka's whole design is built around *high-throughput, replayable, ordered streams* read by *consumer groups that progress through offsets*. Request-reply is *low-latency, point-to-point, one reply matched to one request, reply consumed once and never again*. These are opposite workloads. Kafka's offset-commit model means a consumer reads forward through a partition; it does not "pick out" the one message it wants and leave the rest. Replies have no inherent order relative to requests, but Kafka delivers them in partition order. The reply you want might be behind a thousand replies for other clients in the same partition, and you have to read through all of them to reach it. The correlation-id match still works — you filter by it — but you are filtering a firehose, not pulling a specific message from a queue. And Kafka's strength, replay, is actively unhelpful here: you never want to replay an RPC reply, because the request it answered is long gone.

| Concern | RabbitMQ (queue) | Kafka (log) |
| --- | --- | --- |
| Per-client reply destination | Cheap private queue or direct reply-to | No native concept; topic-per-client is heavy |
| Reply-to / correlation-id | Standard AMQP properties | Roll your own in headers |
| Matching a reply | Pull from your private queue, lookup | Read through partition, filter by corr-id |
| Reply consumed once | Native (ack and gone) | Offsets advance; reply stays in the log |
| Right tool for RPC | Yes | Generally no |

The honest conclusion: **do not do request-reply on Kafka unless you have a specific reason Kafka must be the transport.** Kafka is superb for [event streaming and durable logs](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models); it is the wrong substrate for synchronous-style RPC. If your architecture is Kafka-centric and you need a request/response, the usual right answer is to make a direct RPC (gRPC/HTTP) for that interaction and keep Kafka for the event flows, rather than contorting the log into a reply channel. When teams insist on request-reply over Kafka, it is almost always because "we already have Kafka and do not want another system," and they pay for that convenience with significant complexity in the reply-routing layer. Sometimes that trade is worth it; usually it is not.

## 8. Scatter-gather: fan out and aggregate

Request-reply has a powerful generalization. Instead of one request to one server and one reply, you send one logical request that **fans out to N workers**, each does a piece, and you **aggregate** their replies into one answer. This is scatter-gather, and it is where request-reply over messaging earns its keep, because the broker's native fan-out makes the scatter trivial.

![A directed graph showing a client sending one request that fans out to three workers operating on different shards, each worker sending its reply to an aggregator that waits for a quorum and then produces a merged or partial reply](/imgs/blogs/request-reply-over-messaging-correlation-ids-7.webp)

The canonical example is a sharded search or a sharded aggregate. A user asks "how many active sessions across all regions?" and the data is sharded across five regional services. You publish one request that fans out — via a fanout exchange in RabbitMQ, or by writing to N partitions/topics — so all five shards get it. Each shard computes its local count and replies, all stamped with the *same* correlation id (because it is all one logical request) plus a *shard id* so the aggregator can tell the replies apart. An aggregator collects replies for that correlation id until it has all five (or a quorum, or a deadline), sums them, and produces the final answer. The figure shows exactly this: one request scatters to workers on different shards, and an aggregator gathers the correlated replies.

The aggregator's logic is the interesting part, because it has to decide *when the gather is done*. There are three completion strategies, and you pick based on your latency-versus-completeness needs:

- **Wait for all N.** The simplest: the answer is complete only when every shard has replied. The downside is that the slowest shard determines your latency — one slow or dead shard makes the whole request hang until the timeout. This is the right choice when correctness requires every shard's contribution and a partial answer is wrong.
- **Wait for a quorum (M of N).** Reply when M of N shards have answered, accepting that the answer is approximate. This bounds latency by the M-th fastest shard, not the slowest, which is a huge win when one slow shard would otherwise dominate. Used heavily in search, where "results from 9 of 10 shards" is a perfectly good answer delivered fast.
- **Wait until a deadline, take what you have.** Set a timer; when it fires, aggregate whatever replied and mark the answer partial. This bounds worst-case latency hard, at the cost of sometimes returning incomplete results. Good for dashboards and best-effort aggregates.

```python
class ScatterGather:
    def __init__(self, n_shards, quorum=None, timeout=2.0):
        self.expected = n_shards
        self.quorum = quorum or n_shards          # default: wait for all
        self.timeout = timeout

    async def request(self, body):
        corr_id = str(uuid.uuid4())
        future = loop.create_future()
        self.pending[corr_id] = {
            "replies": {}, "future": future,        # shard_id -> reply
        }
        publish_fanout(body, corr_id)               # scatter to all shards
        try:
            return await asyncio.wait_for(future, self.timeout)
        except asyncio.TimeoutError:
            # deadline hit: return partial with whatever arrived
            return self._aggregate(corr_id, partial=True)

    def _on_reply(self, props, body):
        entry = self.pending.get(props.correlation_id)
        if not entry:
            return
        entry["replies"][props.headers["shard_id"]] = body
        if len(entry["replies"]) >= self.quorum:    # quorum reached
            result = self._aggregate(props.correlation_id, partial=False)
            entry["future"].set_result(result)
```

The pending-request map grows a dimension here: each entry holds not one reply but a *set* of replies keyed by shard id, and the completion check is "do I have enough?" rather than "did the one reply arrive?" But the bones are identical to plain request-reply — a correlation id, a pending map, a deadline. Scatter-gather is request-reply with a many-to-one gather grafted onto the reply side.

#### Worked example: tail latency, all-N versus quorum

Suppose you scatter to 10 shards, and each shard's reply latency is independent with a p50 of 20 milliseconds and a p99 of 200 milliseconds (one slow shard in a hundred). If you wait for all 10, your request's latency is the *maximum* of 10 independent latencies. The probability that *at least one* of 10 shards hits its p99 tail is roughly 1 minus 0.99 to the tenth power, about 9.6 percent — so nearly one in ten of your aggregate requests will be dragged out to ~200 milliseconds by its slowest shard, even though each individual shard is fast 99 percent of the time. Tail latency *amplifies* under fan-out: the more shards you wait for, the more likely one of them is slow, and waiting for all of them means you inherit the worst. Now switch to a quorum of 8 of 10. You no longer wait for the 2 slowest shards, so a single slow shard (or even two) no longer affects you at all — you complete as soon as the 8th-fastest replies, which is far more likely to be near the p50. The quorum trades a small amount of completeness for a dramatic cut in tail latency. This is why every large-scale scatter-gather system — search engines, distributed databases doing scatter reads — uses a quorum or a deadline rather than waiting for every shard. Waiting for all N means your p99 is the p99 of your slowest component, multiplied by your fan-out's chance of hitting it.

## 9. When request-reply over MQ is the wrong tool

We have spent eight sections building the pattern. This section is the counterweight, and it is the most important one to internalize, because the failure mode of this pattern is using it where a direct call belongs.

![A decision matrix comparing direct RPC against request-reply over a message queue across latency, coupling, resilience, and complexity, showing direct RPC wins on latency and simplicity while request-reply over MQ wins on coupling and resilience](/imgs/blogs/request-reply-over-messaging-correlation-ids-6.webp)

The matrix lays out the trade with no spin. Direct RPC wins on **latency** (no broker hops) and **complexity** (a typed call with built-in timeout and retry, no correlation id, no pending map, no reply queue). Request-reply over a queue wins on **coupling** (caller knows only a queue name, not a host) and **resilience** (the queue buffers across a brief server outage and levels load under spikes). You are choosing which pair of properties matters more for *this specific interaction*. There is no universally right answer; there is a right answer per call site.

The clearest signal that you have reached for the wrong tool is **latency sensitivity with no decoupling need**. If a user is staring at a spinner waiting for this exact response, and the two services are deployed together, owned by one team, and scaled together, then the broker buys you nothing and costs you two hops and a pile of machinery. Make the direct call. The broker's decoupling is wasted on services that are already coupled.

The second signal is **the operation has no natural reply**. A startling amount of "request-reply" in the wild is request-reply for an operation that should have been fire-and-forget. The caller does not actually need the result; it just wants confirmation the work was accepted, or it wants the work done and does not care about the outcome inline. If that is the case, you do not need request-reply at all — you need to publish an event and move on, and let the result flow as a *separate* event later if anyone cares.

![A two-panel comparison: request-reply commits the caller to a return trip with a reply queue, a pending map, and a timeout if no reply comes, versus a fire-and-forget event that publishes once with no reply path, no wait, and the caller moving on fully decoupled](/imgs/blogs/request-reply-over-messaging-correlation-ids-9.webp)

The contrast in the figure is the whole decision in two panels. Request-reply (left) commits you to a return trip: a reply queue, a pending-request map, and a timeout you must handle when no reply comes. Fire-and-forget (right) publishes one event and the caller is done — no reply path, no wait, no pending state, fully decoupled. Every piece of machinery in this post — correlation ids, reply queues, pending maps, timeout sweeps — exists *only* because you demanded a reply. If you do not actually need the reply inline, all of that machinery is pure cost with no benefit, and the right design is an event, not a request. Before you build request-reply, ask hard: do I genuinely need this answer *now*, synchronously, to proceed? If the honest answer is "no, I just need the work to happen," publish an event and delete the reply queue from your design.

The third signal is **you are on Kafka and fighting it**, which we covered: if your transport is a log and you are building topic-per-client routing to fake reply queues, step back and ask whether a direct RPC for this one interaction, alongside your Kafka event flows, is simpler. It almost always is.

### When it genuinely is the right tool

To be fair to the pattern, here is when it shines. A request that takes seconds to compute, where the caller is a backend service (not a user-facing spinner) that can tolerate latency, where the worker fleet scales independently and is redeployed often, and where load spikes would otherwise overwhelm the workers — that is request-reply's home turf. Think of a rendering job, a report generation, a heavy computation farmed to a worker pool, an ML inference request to a fleet that autoscales. The caller submits, the queue levels the load, the workers drain at their own pace, the reply comes back via correlation id, and a timeout guards against the worker that died. Here the decoupling and load leveling are worth real money, the latency cost is acceptable because the work is slow anyway, and the complexity is justified by the operational flexibility. That is the pattern doing its job.

## Case studies and war stories

### The reply queue that ate the broker

A team built RPC over RabbitMQ following a tutorial that created an exclusive reply queue *per request*. In development at ten requests per second, it worked flawlessly. In production at peak, the service hit roughly 1,200 requests per second, which meant 1,200 queue declares and 1,200 queue deletes per second hammering the broker's metadata layer. The broker's CPU climbed, the management plane slowed, and queue operations started backing up. The visible symptom was bizarre: RPC *latency* spiked, but the workers were idle. The latency was not in the work — it was in the broker spending all its time creating and tearing down queues. The fix was a two-line change: create one reply queue per client process at startup and reuse it, or better, switch to direct reply-to. Metadata churn dropped to near zero and latency returned to baseline. The lesson: the reply queue's *lifecycle* is the thing that scales or does not, and per-request queue creation is a self-inflicted denial of service.

### The correlation id that matched the wrong request

A service used a monotonic integer counter for correlation ids instead of a UUID, reasoning that within a single process a counter is unique and cheaper. It was — until the process was redeployed. The new process started its counter at 1. But the *old* process had in-flight requests, and some of their replies were still in the shared reply queue. The new process consumed a reply for correlation id 5 — a reply meant for the *old* process's request 5 — found that id 5 was indeed pending in *its* map (a brand-new, unrelated request 5), and resolved the wrong future with the wrong data. The bug manifested as occasional, unreproducible data corruption right after deploys: one user saw another user's data, exactly once, then it vanished. It took weeks to find because it only happened in the narrow window after a restart while old replies were still draining. A UUID would have made it impossible. The lesson: correlation ids cross restart boundaries through the broker's buffers, so they must be unique across restarts, which a counter is not and a UUID is for free.

### The scatter-gather that hung on a dead shard

A search service used scatter-gather across eight shards and waited for all eight before responding. One shard's host had a failing disk and stopped replying — not crashing, just hanging. Every search request fanned out to all eight, got seven fast replies, and then waited for the eighth that never came, all the way to the 5-second timeout, before returning a partial result. User-facing search latency went from 40 milliseconds to 5 seconds for *every query*, because every query waited on the dead shard. The system was technically "working" — it returned correct partial results after the timeout — but it was unusable. The fix was to switch from wait-for-all to a quorum of seven of eight, so a single dead shard no longer gated any request. Latency snapped back to 40 milliseconds and the dead shard's absence was invisible to users. The lesson: in scatter-gather, waiting for all N makes your latency hostage to your slowest component; a quorum or deadline is not an optimization, it is a survival requirement.

### The Kafka RPC that nobody could debug

A team standardized on Kafka for everything and built request-reply on top of it with a shared reply topic that every client read in full, each filtering for its own correlation ids. At low volume it worked. As clients multiplied, every client was reading every reply for every other client — bandwidth on the reply topic grew quadratically with the number of clients, consumers fell behind, replies arrived late, timeouts fired, and retries doubled the request load, which produced more replies, which made the consumers fall further behind. It was a slow-motion meltdown that looked like a capacity problem but was a design problem. They eventually moved those interactions to direct gRPC and kept Kafka for the genuine event streams. The lesson: Kafka's lack of per-client routing makes request-reply scale badly with client count, and "we already have Kafka" is not a good enough reason to build RPC on a log.

## When to reach for this (and when not to)

Reach for request-reply over a message queue when **all** of these hold: the caller is a backend service that tolerates added latency (not a user-facing hot path measured in single-digit milliseconds), the work is slow enough that broker hops are a small fraction of total time, the worker fleet benefits from independent scaling and load leveling, and you genuinely need the answer inline to proceed. Rendering farms, report generators, heavy compute pools, autoscaling inference fleets — these are the canonical fits. Use RabbitMQ with direct reply-to as your default transport; it makes the pattern cheap and correct.

Do **not** reach for it when the call is latency-sensitive with no decoupling benefit (just make the direct RPC), when the operation has no real reply and should be a fire-and-forget event, when the two services are co-owned and co-deployed so the broker's location transparency is wasted, or when your only transport is Kafka and you would be building topic-per-client reply routing to fake what RabbitMQ gives natively. In all of those, the machinery costs more than it returns.

And always — *always* — design the missing reply first. A request-reply client without a timeout is not a request-reply client; it is a memory leak with a happy path. The timeout, the pending-map cleanup, and the graceful handling of late and duplicate replies are not edge cases you bolt on later. They are the core of the pattern. The happy path writes itself; the absent reply is the engineering.

## Key takeaways

- **Request-reply over a queue is a synchronous illusion on asynchronous transport.** The broker stays dumb; the synchrony lives in your client via a correlation id and a pending-request map. You do it for decoupling, load leveling, and resilience — never for speed.
- **The correlation id ties one reply to one request** across an unordered, multiplexed channel. Use a UUID, not a counter — counters collide across processes and restarts and silently match the wrong request.
- **Pick the right reply-queue lifecycle.** One reply queue per client *process* (created at startup, shared across requests) is the sweet spot. One queue per *request* churns the broker's metadata and can DoS it. On RabbitMQ, prefer direct reply-to and manage no queue at all.
- **The pending-request map keys corr-id to a future and a deadline.** Register before you publish, clean up on every exit path (success, timeout, exception), and guard against resolving an already-completed future.
- **No reply is the common case, not an exception.** Arm a timeout on every request, size it from observed end-to-end p99 (which includes queue depth, not just processing time), and use a single periodic sweep instead of per-request timers at high concurrency.
- **Retries create duplicate replies.** A timeout does not mean the work did not happen. Make operations idempotent and handle late, unmatched replies by dropping them gracefully and counting them.
- **Kafka is the wrong substrate for request-reply.** A log has no per-client reply routing, no native reply-to, and replies you must filter from a firehose. If you are Kafka-centric, use a direct RPC for the request/response and keep Kafka for events.
- **Scatter-gather is request-reply with a gather.** Fan out with the same correlation id plus a shard id, and never wait for all N — use a quorum or a deadline, or your tail latency becomes hostage to your slowest shard.
- **The wrong-tool test:** if the call is latency-sensitive with no decoupling need, or the operation has no natural reply, you want a direct RPC or a fire-and-forget event, not request-reply over a broker.

## Further reading

- [Queue vs Pub/Sub vs Log: three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) — why a queue makes request-reply natural and a log makes it awkward.
- [Push vs Pull, acknowledgements, and how consumers read](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) — the delivery mechanics underneath the reply path.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — operating the broker your reply queues live on.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the log model and why it resists per-client reply routing.
- RabbitMQ RPC tutorial and the direct reply-to documentation — the canonical reference for `reply_to`, `correlation_id`, and the `amq.rabbitmq.reply-to` pseudo-queue.
- Hohpe and Woolf, *Enterprise Integration Patterns* — the original catalog entries for Request-Reply, Correlation Identifier, Return Address, and Scatter-Gather.
