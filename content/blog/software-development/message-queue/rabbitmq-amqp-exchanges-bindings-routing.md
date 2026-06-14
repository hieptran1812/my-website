---
title: "RabbitMQ Deep Dive, Part 1: AMQP, Exchanges, Bindings, and Routing"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn the AMQP 0-9-1 model from first principles: why producers never publish to queues, how exchanges and bindings route messages, the four exchange types in depth, connections versus channels, the nameless default exchange, and a full end-to-end routing example you can trace by hand."
tags:
  [
    "message-queue",
    "rabbitmq",
    "amqp",
    "exchanges",
    "routing",
    "kafka",
    "distributed-systems",
    "event-driven",
    "messaging-patterns",
    "pub-sub",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-1.webp"
---

Here is a fact that trips up nearly every engineer the first time they read RabbitMQ code carefully: a producer in AMQP never sends a message to a queue. It cannot. There is no `send_to_queue` call in the protocol. The producer hands a message to an *exchange* along with a short string called a routing key, and then it walks away. What happens next — which queue or queues the message lands in, or whether it lands anywhere at all — is decided entirely by *bindings*, which are little routing rules that live between exchanges and queues. The producer does not know how many queues exist. It does not know how many consumers are attached. It does not even know whether anyone is listening. That indirection is not an accident or an inconvenience to be worked around. It is the single design decision that makes RabbitMQ a *router* rather than a *log*, and it is the reason RabbitMQ exists as a separate category of software at all.

If you have only ever used a simple work queue — push a job in one end, pull it out the other — this indirection looks like pointless ceremony. Why route through an exchange when you could just name the queue? The answer is that the indirection buys you something enormous: the producer and the consumer topology become independent. You can add a second consumer that *also* wants every order event, or a third that only wants European orders, or a fourth that only wants failed payments, and you do this by declaring new queues and new bindings on the broker. The publishing code never changes. Nobody redeploys the producer. That is the whole game, and once you see it, a large fraction of RabbitMQ's apparent complexity resolves into a small set of clean rules.

This post is the protocol and model deep-dive. By the end you will be able to look at any AMQP topology and predict, deterministically, which queues a given message reaches; you will know exactly when to reach for a direct, fanout, topic, or headers exchange; you will understand why you reuse a small number of TCP connections and open many cheap channels on top of them; and you will be able to design a multi-axis routing scheme — by region, by severity, by tenant — and trace a single message through it by hand. The figure below is the entire object model on one canvas: a producer addresses an exchange, the exchange consults its bindings, the bindings select a set of queues, and consumers read from those queues.

![A grid showing the AMQP object model with producer, exchange, bindings, queues, matched set, and consumers connected in a flow](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-1.webp)

This is Part 1 of a two-part RabbitMQ deep dive. Here we cover the *routing* model — the static topology of exchanges, bindings, and queues, and the four ways a message can be matched to a destination. Part 2, [acks, publisher confirms, durability, and quorum queues](/blog/software-development/message-queue/rabbitmq-acks-confirms-durability-quorum-queues), covers the *reliability* model — what happens to a message in flight, how you guarantee it is not lost, and how the queue itself survives a broker crash. This post also complements, rather than repeats, the operations-focused [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) post, which goes deep on clustering, federation, and scaling patterns. If you have not yet internalized the three fundamental messaging shapes, read [Queue vs Pub/Sub vs Log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) first — RabbitMQ is the broker that lets you build *all three* shapes out of the same routing primitives, which is exactly why understanding its routing model is worth a long post.

## 1. Why AMQP separates publishing from routing

Start with the question the AMQP designers were actually answering. The naive design — the one almost everyone reaches for before they know better — is to let a producer name a queue and drop a message into it. That is the model of a Unix pipe, a Go channel, a Python `queue.Queue`. It is simple, it is intuitive, and it is exactly wrong for a system that has to evolve.

Consider what happens the moment a second team wants the same data. Suppose your order service publishes "order placed" events into a queue named `orders`, and a fulfillment worker drains that queue. Now the analytics team shows up and says: we also want every order event, to update a real-time dashboard. With the naive model, you have two bad options. Option one: the analytics consumer reads from the same `orders` queue — but a queue delivers each message to exactly *one* consumer, so now fulfillment and analytics are *stealing* messages from each other, each seeing roughly half the orders. Option two: the producer publishes the message *twice*, once into `orders` and once into `orders_analytics` — which means the producer now knows about analytics, and tomorrow it will have to know about the fraud team, and the warehouse team, and the email team. The producer becomes a directory of every downstream consumer in the company. Every new subscriber is a producer code change and a redeploy.

AMQP cuts this knot by inserting a layer of indirection. The producer publishes to a *named exchange* — say, an exchange called `orders` — with a routing key like `order.placed`. The exchange is not a buffer; it stores nothing. It is a *function* that takes the routing key and the message headers and returns a set of queues. Bindings are the rows of that function's lookup table: "queue `fulfillment` is bound to exchange `orders` with key `order.placed`," "queue `analytics` is bound to exchange `orders` with key `order.placed`," and so on. When the analytics team arrives, they declare their own queue and their own binding. Zero producer changes. The producer has been publishing `order.placed` to the `orders` exchange the whole time; it simply did not care, and still does not care, who is bound.

### The exchange is a routing function, not a mailbox

The mental shift that unlocks everything is this: an exchange holds no messages. People new to RabbitMQ often picture the exchange as a place where messages pile up. It is not. The instant a message arrives at an exchange, the exchange evaluates its bindings and either copies the message into the matching queues or — if nothing matches — drops it (or routes it to an alternate exchange, which we will get to). The exchange has no memory of the message a microsecond later. All buffering, all retention, all the "messages waiting to be processed" lives in *queues*. The exchange is pure routing logic; the queue is the only stateful thing.

This separation is why I keep saying RabbitMQ is a *router* and not a *log*. A Kafka topic is a durable, ordered, retained sequence of bytes that consumers read by advancing an offset — the storage *is* the model you program against, as covered in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log). RabbitMQ inverts that: the storage (the queue) is dumb and the *routing* is the smart, expressive part. You get topic patterns, header matching, fanout, and per-binding rules — a small routing language — instead of a partition number and an offset. The cost is that RabbitMQ deletes messages after they are acknowledged, so there is no replay. The benefit is that the routing is vastly more flexible than "pick a partition."

### Smart broker, dumb consumer

One more framing before we go to the wire. RabbitMQ is a *smart broker, dumb consumer* system, the mirror image of Kafka's *dumb broker, smart consumer*. The RabbitMQ broker knows about routing, bindings, retries, dead-lettering, per-message expiry, priorities, and per-consumer flow control. The consumer's job is narrow: pull a message, do the work, send an acknowledgement. Because the broker is doing the thinking, the routing model — the subject of this post — is where the design effort goes. Get the exchange topology right and your consumers stay simple. Get it wrong and you will find yourself writing routing logic *inside* your consumers, which is the smell that tells you the topology is fighting you.

## 2. Connections and channels (multiplexing)

Before a producer can publish anything, it has to talk to the broker, and AMQP has a two-level connection model that confuses people and causes a startling fraction of real production incidents. There are **connections** and there are **channels**, and they are not the same thing. Understanding the difference is not optional trivia — it is the difference between a client that scales to thousands of publishers on one box and a client that exhausts the broker's file descriptors at a few hundred.

A **connection** is a single TCP connection (usually wrapped in TLS) from your client to the broker. It costs a socket, a file descriptor on both ends, a TLS handshake, and a chunk of memory on the broker for buffers and heartbeat tracking. Connections are *expensive* and relatively *slow* to open — the TLS handshake alone is a couple of round trips. You want few of them, and you want them long-lived.

A **channel** is a lightweight, virtual connection *inside* a TCP connection. AMQP multiplexes many channels over one socket by tagging every frame on the wire with a channel number. A channel is where almost all the real work happens: you declare exchanges and queues on a channel, you publish on a channel, you consume on a channel, you acknowledge on a channel. Opening a channel is cheap — it is a single round trip, no handshake, a few kilobytes of broker memory. The figure below shows the layering: one TCP connection at the bottom, many channels riding on top of it, each channel publishing into exchanges that feed queues.

![A stack diagram showing one TCP connection at the base, many multiplexed channels above it, then exchange and queue layers](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-6.webp)

### Why you reuse connections and open many channels

The rule that falls out of this is blunt: **open a small number of long-lived connections and a larger number of channels on top of them.** A common production shape is one connection per process (or one for publishing and one for consuming, since you want them isolated), with one channel per thread or per logical task. The reason to separate publishing and consuming onto different connections is *flow control*: when a consumer falls behind and the broker pushes back, it can throttle the consuming connection without strangling your publishers, and vice versa.

The anti-pattern — the one that shows up in incident postmortems — is opening a brand-new connection for *every message* or *every request*. I have watched a service do exactly this: a web handler that, on each HTTP request, opened a fresh AMQP connection, published one message, and closed it. At low traffic it worked fine. At a few hundred requests per second it melted: each connection meant a TLS handshake (two round trips of latency added to every request), a file descriptor churned on the broker, and a spike of broker CPU doing connection setup and teardown. The broker spent more time managing connection lifecycle than routing messages. The fix was a single shared, long-lived connection with a small pool of channels, and the broker's CPU dropped by an order of magnitude.

### Channels are not thread-safe

There is a sharp edge here that the docs state but people ignore: a channel is **not** safe to share across threads. The AMQP frame multiplexing assumes a single writer per channel; two threads publishing on the same channel will interleave frames and corrupt the protocol stream, and you will get baffling errors. The rule is one channel per thread. If you have a thread pool, give each thread its own channel (or a small per-thread channel pool). Connections *are* safe to share across threads — that is the whole point of multiplexing — but channels are not. Mix this up and you get intermittent `UNEXPECTED_FRAME` errors that are maddening to debug because they depend on timing.

Here is the canonical Python shape using `pika`, showing one connection reused across publishes with a single channel:

```python
import pika

# One long-lived connection, reused across many publishes.
# In a real service this lives for the process lifetime, not per-request.
params = pika.ConnectionParameters(
    host="rabbitmq.internal",
    heartbeat=30,          # detect dead peers within 30s
    blocked_connection_timeout=300,
)
connection = pika.BlockingConnection(params)

# A channel is cheap. One per thread; never share a channel across threads.
channel = connection.channel()

# Declare the topology once (idempotent — safe to call every startup).
channel.exchange_declare(exchange="orders", exchange_type="topic", durable=True)

# Publish many messages on the SAME channel and SAME connection.
for order_id in range(1000):
    channel.basic_publish(
        exchange="orders",
        routing_key="order.placed.eu",
        body=f"order {order_id}".encode(),
        properties=pika.BasicProperties(delivery_mode=2),  # persistent message
    )

connection.close()  # only at shutdown, not after each publish
```

#### Worked example: 1 connection with 50 channels vs 50 connections

Let us put numbers on the connection-versus-channel decision, because the difference is not subtle. Suppose you have a service with 50 worker threads, each of which needs to publish to RabbitMQ. You have two designs.

**Design A — 50 connections, one per thread.** Each connection is a TCP socket plus a TLS session. On the broker side, RabbitMQ allocates roughly 100 KB to a few hundred KB of memory per connection for read and write buffers, the Erlang processes that manage the connection and its reader/writer, and heartbeat state. Call it ~150 KB per connection as a working estimate (it varies with buffer tuning and TLS). Fifty connections is therefore on the order of 7.5 MB of broker memory just for connection overhead, 50 file descriptors on the broker, 50 TLS handshakes at startup (and again on every reconnect storm), and 50 sockets your client also has to track. If 200 such service instances connect, you are at 10,000 connections — and RabbitMQ's default file-descriptor and connection limits start to bite. Connection churn during a deploy (all 200 instances reconnecting at once) can knock a node over with a setup-CPU spike.

**Design B — 1 connection, 50 channels.** One TCP socket, one TLS handshake, ~150 KB of connection overhead, plus 50 channels at roughly a few KB of broker memory each — call it 4 KB per channel, so ~200 KB for the channels. Total broker memory: under 400 KB versus 7.5 MB. One file descriptor versus 50. One TLS handshake versus 50. At 200 instances you are at 200 connections and 10,000 channels — channels are cheap, and 10,000 of them is a memory and bookkeeping non-event compared to 10,000 connections.

The throughput story is the same direction. A single TCP connection on a modern network can comfortably carry hundreds of thousands of small messages per second — you are not bandwidth-bound on the socket at typical RabbitMQ message rates (RabbitMQ tops out in the tens of thousands to low hundreds of thousands of messages per second per node well before one TCP connection saturates). So 50 channels on one connection do not contend for socket bandwidth in any way you will notice. What 50 *connections* buy you is 50 separate TLS/connection-setup costs and 50× the broker connection memory, for no throughput benefit. The only reason to split into a *few* connections is isolation — separating publishers from consumers, or isolating a noisy tenant — not raw scale. The rule: scale with channels, not connections. Use a handful of connections per process for isolation; open as many channels as you have concurrent logical streams.

There is one caveat worth stating: channels on the same connection *do* share the single socket's head-of-line ordering, so a very large message on one channel can briefly delay frames on another channel sharing that connection. In practice this matters only if you mix tiny control messages with multi-megabyte payloads on the same connection; the fix is to give large-payload traffic its own connection. That is the second legitimate reason to open more than one connection.

### What actually travels on the wire

It helps to know what a channel actually carries, because the wire-level truth is simpler than people fear. AMQP 0-9-1 is a *frame*-based binary protocol. Every frame on the connection is tagged with a channel number (channel 0 is reserved for connection-level housekeeping; your channels are numbered 1 and up). There are four frame types: **method frames** (the commands — `exchange.declare`, `queue.bind`, `basic.publish`, `basic.consume`), **content header frames** (the message properties and body size that follow a publish), **content body frames** (the actual payload bytes, split across multiple frames if the message is large), and **heartbeat frames** (keepalives that let each side detect a dead peer). When you call `basic_publish`, the client emits a `basic.publish` method frame naming the exchange and routing key, then a content header frame with the message properties, then one or more body frames with the payload. The broker reassembles these, evaluates the exchange's routing, and the message is on its way.

The reason this matters operationally: because frames from different channels interleave on one socket, the broker tags and demultiplexes them by channel number, which is exactly why a single channel must have a single writer. It is also why heartbeats are per-*connection*, not per-channel — a dead TCP connection takes all its channels down with it. Set the heartbeat interval (the `heartbeat=30` in the code above) low enough that you detect a silently dropped connection in seconds rather than waiting for a TCP timeout that can be minutes. A too-high heartbeat is a classic cause of "the consumer looks connected but never receives anything" — the socket died, nobody noticed, and the channels on it are zombies.

## 3. The exchange-binding-queue model

Now that the client can talk to the broker over channels, let us nail down the three nouns that make up every AMQP topology: **exchanges**, **queues**, and **bindings**. These are the only structural objects in the routing model, and everything else is configuration on top of them.

An **exchange** is a named routing function. You give it a name (`orders`, `logs`, `events`) and a type (`direct`, `fanout`, `topic`, or `headers`), and from then on it routes every message published to it according to the rules of its type. Exchanges are typically declared `durable` so they survive a broker restart, but remember — they store nothing, so "durable exchange" only means the *declaration* survives a restart, not any messages.

A **queue** is a named, ordered buffer that holds messages until a consumer acknowledges them. Queues are where messages actually live. A queue has properties that matter enormously for reliability — durable, exclusive, auto-delete, TTL, dead-letter target — but those are Part 2's subject. For routing, the only thing that matters is that a queue is a destination that can be bound to exchanges.

A **binding** is a rule that connects an exchange to a queue, with an optional **binding key** (and, for headers exchanges, a set of header-match arguments). A binding says "messages arriving at exchange `X` that match key `Y` should be copied into queue `Z`." Bindings live on the broker and are declared by whoever owns the queue — typically the consumer. This is the crucial inversion: **the consumer declares the binding, so the consumer decides what it receives, and the producer never has to know.**

### The before-and-after of indirection

The figure below makes the payoff concrete by contrasting the two models side by side. On the left, the wrong model: the producer publishes straight to a named queue, which hard-wires a single destination and means every new consumer is a producer code change. On the right, the AMQP model: the producer publishes to an exchange with a key, bindings select the queues, and a new consumer is a pure broker-side operation — declare a queue, add a binding, done, with no redeploy of anything that publishes.

![A before-and-after diagram contrasting publishing directly to a queue against publishing to an exchange where bindings select queues](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-7.webp)

This is worth dwelling on because it is the entire architectural argument for RabbitMQ over a naive queue. The indirection is not free — you have to declare exchanges and bindings, you have to choose an exchange type, you have to design a routing-key scheme — but what you buy is *topology that evolves without touching producers*. In a large organization with many teams reading the same event streams, this is the difference between a system that ossifies and one that grows. The producer publishes a well-defined event to a well-defined exchange with a well-defined routing key, and that contract is stable. Consumers come and go on the other side of the exchange, and the producer is blissfully unaware.

### Declaring the topology in code

Here is what declaring a small topology looks like in Python. Note that all of these declarations are *idempotent* — calling them again with the same arguments is a no-op, which is why services declare their topology on every startup rather than assuming someone set it up:

```python
# Consumer-side topology declaration. The producer never runs this.
channel.exchange_declare(exchange="orders", exchange_type="topic", durable=True)

# I want a durable queue for the fulfillment workers.
channel.queue_declare(queue="fulfillment", durable=True)

# Bind it: deliver every order.placed.* message into my queue.
channel.queue_bind(
    exchange="orders",
    queue="fulfillment",
    routing_key="order.placed.*",
)

# A second team binds their own queue with a DIFFERENT key — no producer change.
channel.queue_declare(queue="eu_analytics", durable=True)
channel.queue_bind(
    exchange="orders",
    queue="eu_analytics",
    routing_key="order.placed.eu",
)
```

Two consumers, two queues, two bindings, one exchange, and the producer that publishes `order.placed.eu` to `orders` has no idea either of them exists. That is the model. The next four sections are just the four different *types* of exchange — the four different routing functions you can put in the middle.

## 4. Direct exchanges: exact routing-key match

The **direct** exchange is the simplest routing function and the right default for point-to-point work distribution. Its rule is exact string equality: a message published with routing key `K` is delivered to every queue bound to the exchange with binding key exactly equal to `K`. No wildcards, no patterns — just `==`.

The figure below shows the shape. A publisher sends a message with routing key `pdf`. Three queues are bound to the direct exchange with binding keys `pdf`, `image`, and `video` respectively. Only the `pdf`-bound queue matches, so only it receives the message; the other two are not touched.

![A graph showing a direct exchange delivering a pdf-keyed message only to the queue bound with binding key pdf](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-2.webp)

### Why "direct" is the workhorse

Direct exchanges are how you build classic task queues with multiple task types on one exchange. Imagine a media-processing pipeline: you publish `pdf` jobs, `image` jobs, and `video` jobs to a single `tasks` direct exchange, and you bind a dedicated pool of workers to each binding key. PDF workers bind with key `pdf`, image workers bind with key `image`. A `pdf` message goes only to PDF workers; the routing key is doing the dispatch. This keeps your workers specialized and your exchange topology readable.

It is worth noting that *multiple queues can share the same binding key*. If two different queues both bind to the direct exchange with key `pdf`, then a `pdf` message is copied into *both* — direct does not mean "exactly one queue," it means "every queue whose binding key matches exactly." So you can fan out with a direct exchange too, as long as the queues share a key. Conversely, *one queue can have multiple bindings* with different keys: bind a queue with both `pdf` and `image` and it receives both kinds. Routing is the union of all matching bindings, evaluated independently.

### The competing-consumers pattern

Inside a single queue, multiple consumers compete: the broker hands each message to one available consumer, round-robin by default. This is the [point-to-point queue model](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models), and it is how you scale a worker pool horizontally — add more consumers to the same queue and they share the load, each message going to exactly one of them. Do not confuse this with fanout: competing consumers on *one* queue split the work; multiple *queues* bound to a fanout exchange each get a full copy. The unit of "exactly once delivery to a worker" is the queue; the unit of "broadcast to everyone" is the binding fan-out across multiple queues. Whether the broker pushes those messages to consumers or consumers pull them, and how acknowledgements close the loop, is the subject of [push vs pull and acknowledgements](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read).

Here is a direct-exchange task dispatcher:

```python
# Producer: publish typed jobs to a direct exchange.
channel.exchange_declare(exchange="tasks", exchange_type="direct", durable=True)

channel.basic_publish(exchange="tasks", routing_key="pdf",   body=b"render report.pdf")
channel.basic_publish(exchange="tasks", routing_key="image", body=b"resize hero.png")

# Consumer (PDF worker): bind only to the 'pdf' key.
channel.queue_declare(queue="pdf_jobs", durable=True)
channel.queue_bind(exchange="tasks", queue="pdf_jobs", routing_key="pdf")
# Image workers would bind "image_jobs" with routing_key="image", etc.
```

### Direct exchanges and reply-to RPC

Direct exchanges (and the default exchange, which is a direct exchange) are also the backbone of request-reply over RabbitMQ. The pattern: a client creates a temporary, exclusive reply queue, publishes a request with two special properties — `reply_to` set to that queue's name and a unique `correlation_id` — and the server, after processing, publishes its response to the default exchange with routing key equal to the `reply_to` queue name. Because the default exchange auto-binds every queue to its own name, that response lands straight back in the client's reply queue, and the client matches it to the original request by `correlation_id`. The whole RPC round trip rides on direct routing-key equality, no topic patterns needed. It works, but be honest about it: RPC over a message broker adds a broker round trip to every call and couples request latency to broker health, so reserve it for cases where you genuinely want the broker's buffering and load-leveling between caller and worker, not as a default replacement for a synchronous HTTP or gRPC call.

When your routing axis is a single, closed set of exact values — a job type, a destination service name, a severity level you match exactly — direct is the correct, fastest, most predictable choice. The moment you find yourself wanting "all keys that start with `order.`" or "all keys ending in `.error`," you have outgrown direct and want topic, which we reach in §6.

## 5. Fanout exchanges: broadcast to all

The **fanout** exchange is the bluntest instrument and sometimes exactly the right one. Its rule is: *ignore the routing key entirely and deliver a copy of every message to every bound queue.* There is no matching, no pattern, no decision. Bind a queue and it gets everything. Unbind it and it gets nothing.

The figure below shows the broadcast. A publisher sends a message; the routing key is ignored; three queues are bound — an audit queue, a cache-invalidation queue, and a search-indexing queue — and all three receive a copy.

![A graph showing a fanout exchange copying one message to three bound queues regardless of routing key](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-3.webp)

### When broadcast is what you actually want

Fanout is the AMQP realization of the pub/sub broadcast model. Use it when an event genuinely concerns *every* subscriber and you do not want to maintain a routing-key scheme. Classic cases: a configuration-changed event that every service instance must react to, a cache-invalidation event that every cache node must apply, a "user updated" event that the search indexer, the audit log, and the recommendation engine all want in full. Each of these consumers binds its own queue to the fanout exchange and receives every event independently — independent queues, independent backlogs, independent failure. If the search indexer falls behind, its queue grows, but the cache-invalidation queue is unaffected. That isolation is the point: each subscriber has its own buffer.

A subtlety that matters: each bound queue is *independent*, so each gets its own copy and its own delivery guarantees. If one consumer is slow, its queue backs up without affecting the others. This is genuinely different from a single shared queue with competing consumers, where one slow consumer just means the others pick up its slack. With fanout into N queues, you have N independent streams of the same data. That is exactly the property you want for "every subscriber sees every event," and exactly the property you do *not* want for "distribute work across a pool" — for the latter, you want one queue and competing consumers.

### Fanout plus per-queue routing

A pattern I like: publish to a fanout exchange, but then have each consumer's queue do *secondary* filtering with a per-message check, or use a fanout for the broadcast tier and bind those queues with TTLs and dead-letter targets for reliability. But honestly, if you find yourself wanting to filter the fanout, you probably want a topic exchange instead, where the filtering is done *in the broker* by the routing key rather than in your consumer. Fanout earns its keep precisely when every subscriber wants *everything* — the moment selectivity enters, climb up to topic.

```python
# Producer: broadcast a config-change event to everyone.
channel.exchange_declare(exchange="config_events", exchange_type="fanout", durable=True)
channel.basic_publish(exchange="config_events", routing_key="", body=b"reload feature flags")
#                                              ^^^^^^^^^^^^^^^ key ignored for fanout

# Each subscriber binds its OWN queue; no routing key needed.
channel.queue_declare(queue="cache_node_7", durable=True)
channel.queue_bind(exchange="config_events", queue="cache_node_7")  # no routing_key
```

Note the empty routing key on publish — fanout ignores it, so by convention you pass `""`. And note that `queue_bind` for fanout takes no meaningful routing key. The binding exists purely to attach the queue to the exchange.

## 6. Topic exchanges: pattern matching with * and #

The **topic** exchange is the most expressive of the four and the one that makes RabbitMQ's routing genuinely powerful. It matches a *dotted* routing key against *patterns* in the binding keys, using two wildcards. This is where you encode multi-dimensional routing — region, severity, tenant, event type — into a single string and let the broker slice it however each consumer wants.

The rules are precise, so learn them exactly. A routing key is a dot-separated list of words, like `logs.eu.error` or `order.placed.us.priority`. A binding key is also dot-separated but may contain two special wildcards:

- `*` (star) matches **exactly one word**.
- `#` (hash) matches **zero or more words**.

So the binding key `logs.*.error` matches `logs.eu.error` and `logs.us.error` but *not* `logs.error` (the `*` demands a word between `logs` and `error`) and *not* `logs.eu.auth.error` (the `*` matches exactly one word, not two). The binding key `logs.#` matches `logs`, `logs.eu`, `logs.eu.error`, and `logs.eu.auth.error.critical` — the `#` swallows any number of trailing words, including zero. And `logs.#.error` matches `logs.error`, `logs.eu.error`, and `logs.eu.auth.error`, because `#` can stand in for zero or more words between `logs` and the final `error`.

The figure below shows a publisher sending `logs.eu.error` into a topic exchange. The errors queue is bound with `logs.#.error` and matches (the `#` covers `eu`). The eu queue is bound with `logs.eu.*` and matches (the `*` covers `error`). The us queue is bound with `logs.us.*` and does not match. So this one message reaches two of the three queues.

![A graph showing a topic exchange routing logs.eu.error to two queues whose wildcard binding keys match and skipping the us queue](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-4.webp)

### Designing a routing-key scheme

The art of topic exchanges is designing the routing key so the dimensions you will want to slice by are *positions* in the dotted key. Put the most-significant axis first if you frequently route on it. A common shape for logging is `logs.<region>.<service>.<severity>`, e.g. `logs.eu.auth.error`. With that scheme, a consumer that wants all EU logs binds `logs.eu.#`; a consumer that wants all errors anywhere binds `logs.#.error`; a consumer that wants EU auth errors specifically binds `logs.eu.auth.error`. The scheme is the API, and once you publish it, consumers compose bindings against it without you lifting a finger.

A few hard-won rules for routing-key schemes:

- **Order axes by selectivity and query frequency.** The leftmost words are the easiest to match on with `*` and the cleanest to prefix-match with `#`. Put the axis you slice on most often near the front.
- **Keep the number of words stable across a given event type.** If `order.placed` events are sometimes three words and sometimes five, your wildcard bindings become guesswork. Pick a fixed arity per event family and stick to it.
- **Never let routing keys exceed 255 bytes** — that is the AMQP limit, and you will not hit it with sane schemes, but generated keys (e.g. embedding a tenant UUID) can creep toward it.
- **Avoid putting high-cardinality values in the key** if you intend to bind on them exactly. Binding one queue per tenant UUID means thousands of bindings; the broker handles it, but topic matching cost grows with binding count.

#### Worked example: multi-region logging with `logs.<region>.<severity>` and tracing `logs.eu.error`

Let us design a real topic topology and trace a message through it by hand, because this is the example that makes the wildcard rules click. We are building a logging fabric for a service running in three regions — `eu`, `us`, and `apac` — emitting logs at severities `info`, `warn`, and `error`. We choose the routing-key scheme `logs.<region>.<severity>`, so example keys are `logs.eu.error`, `logs.us.info`, `logs.apac.warn`.

Now we set up five consumer queues, each with a binding that expresses a different slice of interest:

| Queue | Binding key | What it wants |
| --- | --- | --- |
| `q_all` | `logs.#` | every log line, all regions, all severities |
| `q_errors` | `logs.*.error` | every error, any region (exactly region + error) |
| `q_eu` | `logs.eu.*` | everything from the EU region |
| `q_eu_err` | `logs.eu.error` | EU errors only (exact match) |
| `q_us_warn_err` | `logs.us.warn` plus `logs.us.error` | US warnings and errors (two bindings) |

Now publish a single message with routing key `logs.eu.error` and trace which queues receive it. Evaluate each binding independently:

- `q_all` bound `logs.#` — the `#` matches `eu.error` (two words). **Match.** Receives the message.
- `q_errors` bound `logs.*.error` — the `*` matches `eu` (one word), and the literal `error` matches. **Match.** Receives the message.
- `q_eu` bound `logs.eu.*` — literal `logs`, literal `eu`, and `*` matches `error` (one word). **Match.** Receives the message.
- `q_eu_err` bound `logs.eu.error` — exact match, word for word. **Match.** Receives the message.
- `q_us_warn_err` bound `logs.us.warn` and `logs.us.error` — both require region `us`, but our key has region `eu`. **No match** on either binding. Does not receive the message.

So `logs.eu.error` lands in **four** of the five queues: `q_all`, `q_errors`, `q_eu`, and `q_eu_err`. The broker copies the message into each matching queue independently — four copies, four independent backlogs. If a second message `logs.apac.info` is published, trace it: `q_all` matches (`#`), `q_errors` does not (`info` != `error`), `q_eu` does not (`apac` != `eu`), `q_eu_err` does not, `q_us_warn_err` does not. So `logs.apac.info` lands in exactly one queue, `q_all`.

This is the entire power of topic exchanges in one example: a *single* publish, a *single* routing key, and the broker fans it out to precisely the set of consumers whose interest patterns match — no consumer-side filtering, no producer awareness of who is listening, and new slices added by declaring new bindings. If tomorrow you want "all APAC errors," you add a queue bound `logs.apac.error` and publishers never know.

```python
# Producer: every log line published to one topic exchange.
channel.exchange_declare(exchange="logs", exchange_type="topic", durable=True)
channel.basic_publish(exchange="logs", routing_key="logs.eu.error", body=b"db timeout")

# Consumers each compose a binding pattern against the published scheme.
channel.queue_declare(queue="q_errors", durable=True)
channel.queue_bind(exchange="logs", queue="q_errors", routing_key="logs.*.error")

channel.queue_declare(queue="q_eu", durable=True)
channel.queue_bind(exchange="logs", queue="q_eu", routing_key="logs.eu.*")
```

### Routing-key naming conventions that age well

Because the routing key is a contract that producers publish and consumers bind against, treating it casually is how you end up with a routing scheme nobody can reason about three years in. A few conventions I enforce:

- **Lowercase, dot-separated, no spaces.** `logs.eu.error`, not `Logs.EU.Error`. Case sensitivity in matching means `Error` and `error` are different words, and that inconsistency will bite someone. Pick lowercase and never deviate.
- **Most-significant axis first, descending.** Read the key left to right as "domain, then sub-domain, then detail." `order.placed.eu.flagged` reads as a path from general to specific. This makes prefix matching with `#` natural — `order.#` is "all order events" — and keeps the `*` positions intuitive.
- **A fixed grammar per event family, documented.** Write down that `order` events are always `order.<action>.<region>.<flag>` and enforce it. The grammar *is* the API; an undocumented, drifting key scheme is an API with no spec.
- **Reserve a verb/noun position for the event type.** Distinguish `order.placed` from `order.cancelled` so consumers can bind on the action. Putting the event type in a known position is what lets the fraud queue bind `order.*.*.flagged` and the fulfillment queue bind `order.placed.#` against the same stream.
- **Do not encode high-cardinality identifiers you will bind on exactly.** A tenant UUID in the key is fine if every consumer wants `tenant.#` style prefix matches; it is a problem if you create one binding per tenant, because you then have a binding count equal to your tenant count. For per-entity partitioning, the consistent-hash exchange is the right tool, not thousands of exact bindings.

The discipline pays off when a new consumer arrives. With a documented grammar, they read the spec, write one binding, and they are done — no meeting, no producer change, no archaeology to figure out what keys actually get published. The routing key is the most under-appreciated API surface in a RabbitMQ system; design it like one.

### The performance cost of topic matching

Topic matching is not free. RabbitMQ compiles the set of binding keys on a topic exchange into a trie (a prefix tree of words) and walks it per message, which is efficient, but matching cost still scales with the number of bindings and the structure of the patterns. A topic exchange with tens of thousands of bindings — for instance, one binding per tenant — will spend measurably more CPU per message than a direct exchange with a handful of bindings. For most systems this is irrelevant; at very high message rates with very many bindings, it is a real consideration, and is one reason the [production scaling post](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) discusses the consistent-hash exchange plugin for high-cardinality partitioning instead of one binding per key. The takeaway: topic is expressive and cheap *enough* for normal cardinalities; do not use it as a hash table with a hundred thousand exact bindings.

## 7. Headers exchanges and the default exchange

Two more routing mechanisms round out the model: the **headers** exchange (the rarely-used fourth type) and the **default exchange** (the nameless one that makes the simplest RabbitMQ tutorials work). The figure below is the taxonomy — one root routing function splitting into the four built-in types, with the default exchange shown as a special case of direct.

![A tree showing the exchange routing function branching into direct, fanout, topic, and headers types with the nameless default as a special direct exchange](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-8.webp)

### Headers exchanges: matching on attributes, not the key

A **headers** exchange ignores the routing key and instead matches on the message's *headers* — the arbitrary key-value attributes you attach to a message. A binding on a headers exchange specifies a set of header values to match, plus a special argument `x-match` that is either `all` (every specified header must match — logical AND) or `any` (at least one must match — logical OR).

This is useful when your routing axis is not naturally a single dotted string. Suppose messages carry headers `format: pdf`, `type: report`, `region: eu`. A binding with `x-match: all` and headers `{format: pdf, type: report}` matches only messages that are *both* PDFs *and* reports. A binding with `x-match: any` and `{region: eu, region: us}`... well, headers can only carry one value per key in a binding, so the `any` form is for "match if any of these distinct header keys is present with the given value." The headers exchange shines when you have several independent boolean dimensions and expressing them as ordered positions in a dotted topic key would be awkward or fragile.

In practice, headers exchanges are uncommon. Most routing axes *can* be flattened into a topic key, and topic is faster (trie-based) and more familiar. Reach for headers only when the routing genuinely depends on multiple non-hierarchical attributes that do not compose cleanly into an ordered dotted string — for example, when the same message might be routed by format *or* by content-type *or* by a binary flag, in combinations, and you want `x-match: any`/`all` semantics rather than positional matching. Here is the shape:

```python
channel.exchange_declare(exchange="docs", exchange_type="headers", durable=True)

# Bind: deliver messages that are BOTH pdf AND report (x-match=all).
channel.queue_declare(queue="pdf_reports", durable=True)
channel.queue_bind(
    exchange="docs",
    queue="pdf_reports",
    arguments={"x-match": "all", "format": "pdf", "type": "report"},
)

# Publish with headers; routing key is ignored by headers exchanges.
channel.basic_publish(
    exchange="docs",
    routing_key="",
    body=b"...",
    properties=pika.BasicProperties(headers={"format": "pdf", "type": "report"}),
)
```

### The default exchange and why `basic_publish("", "queuename")` works

Every RabbitMQ broker has a pre-declared, nameless exchange — the **default exchange** — with the empty string `""` as its name. It is a *direct* exchange with a special, automatic property: every queue is *automatically bound* to the default exchange with a binding key equal to the queue's own name. You do not declare these bindings; the broker maintains them for you as queues come and go.

This is the trick behind every "hello world" RabbitMQ tutorial. When you write:

```python
channel.queue_declare(queue="hello", durable=True)
channel.basic_publish(exchange="", routing_key="hello", body=b"first message")
```

you are publishing to the default (nameless) exchange with routing key `hello`. Because the default exchange is a direct exchange and the queue `hello` is automatically bound to it with binding key `hello`, the message routes straight into the `hello` queue. It *looks* like you are publishing directly to a queue — and that is the intended illusion — but you are not. You are publishing to an exchange that happens to have an automatic binding named after every queue. The "publish to a queue" mental model that this convenience creates is exactly the one that confuses people later when they meet real exchanges, which is why I spent §1 dismantling it.

The default exchange is genuinely useful for the simplest case: a single queue, point-to-point, no routing logic, "send this job to that worker pool by name." But the instant you need fan-out, patterns, or topology that evolves, you declare a real named exchange and stop relying on the magic auto-bindings. A good rule: the default exchange is fine for quick scripts and the simplest task queues; production topologies should use explicitly named exchanges so the routing is visible in your declarations and not hidden in a convenience.

#### Worked example: headers vs topic for a multi-attribute filter

Let us decide between a topic and a headers exchange on a concrete case, because the choice is not arbitrary. A document-processing service routes messages by three independent attributes: `format` (one of pdf, docx, html), `priority` (high or low), and `language` (en, fr, de, ...). Consumers want arbitrary combinations: "all high-priority PDFs," "all French documents regardless of format," "all high-priority documents in any format or language."

**Topic approach.** You flatten the attributes into a dotted key, say `doc.<format>.<priority>.<language>`, e.g. `doc.pdf.high.fr`. "All high-priority PDFs" binds `doc.pdf.high.*`. "All French documents" binds `doc.*.*.fr`. "All high-priority anything" binds `doc.*.high.*`. This works, but notice the friction: the *positions* are fixed, so "French documents" must spell out `doc.*.*.fr` with a `*` for every intervening axis, and if you later add a fourth axis the existing bindings silently break because the arity changed. The ordering of axes is baked into every binding.

**Headers approach.** You attach headers `{format: pdf, priority: high, language: fr}` and bind with `x-match` rules. "All high-priority PDFs" binds `{x-match: all, format: pdf, priority: high}`. "All French documents" binds `{x-match: all, language: fr}`. "All high-priority anything" binds `{x-match: all, priority: high}`. There is no positional coupling — each binding names only the attributes it cares about, and adding a fourth attribute breaks nothing, because bindings that do not mention it simply ignore it.

**The decision.** When the routing axes are *independent* (not hierarchical) and you frequently match on *subsets* of them in arbitrary combinations, headers is genuinely cleaner — no `*`-padding, no arity fragility. When the axes have a natural *hierarchy* or *order* (region contains service contains severity), topic is cleaner and faster, because the trie-based matching is more efficient than headers' per-binding attribute comparison and the dotted key reads like a path. For this document case, with three orthogonal attributes matched in arbitrary subsets, headers is the better fit — and this is exactly the narrow situation where the otherwise-rare headers exchange earns its place. Most routing is hierarchical, which is why topic wins most of the time; recognize the orthogonal-attributes case and headers stops looking obscure.

## 8. Routing keys, binding keys, and alternate exchanges

Let us tighten the vocabulary, because "routing key" and "binding key" are used loosely and the distinction matters. A **routing key** is the string the *producer* attaches to a message at publish time. A **binding key** is the string (or pattern) the *consumer* attaches to a *binding* when wiring a queue to an exchange. Routing happens when the exchange compares the message's routing key against each binding's binding key, according to the exchange type's rule (exact for direct, pattern for topic, ignored for fanout, headers-based for headers). Producer sets the routing key; consumer sets the binding key; the exchange does the comparison. Keep those three roles straight and the whole model reads cleanly.

### What happens to unroutable messages

A natural question: what happens when a message matches *no* binding? By default, the exchange silently **drops** it. The publish succeeds from the producer's point of view — the broker accepted the message — but it routes to zero queues and simply vanishes. This is a classic source of "I published it but nobody got it" bugs: a typo in the routing key, or a queue that was never bound, and the message is gone with no error.

There are three ways to stop unroutable messages from vanishing silently, and you should know all three:

1. **The `mandatory` flag.** Publish with `mandatory=true` and if the message is unroutable, the broker returns it to the producer via a `basic.return` callback instead of dropping it. The producer can then log, alert, or store it. This is the "tell me when I sent something nowhere" option. It requires the producer to register a return handler and is most useful for catching misconfiguration in development and for critical messages you cannot afford to lose to a routing typo.

2. **Alternate exchanges (AE).** You can declare an exchange with an `alternate-exchange` argument naming a *second* exchange that receives any message the first exchange could not route. The unroutable message is republished to the alternate exchange, where it can be bound to a catch-all queue. This is the broker-side, producer-agnostic way to catch unroutable messages, and it is my preferred approach for production: route the lost-and-found to a queue that an operator monitors, so nothing vanishes silently and the producer needs no special handling.

3. **A catch-all binding.** On a topic exchange, bind a monitoring queue with `#`, which matches every routing key. Anything routable goes to its proper queue *and* a copy to the catch-all. This catches messages that *did* route but lets you observe everything; it does not specifically capture the *unroutable* ones (those still need `mandatory` or an alternate exchange), so it complements rather than replaces them.

Here is an alternate exchange wired up so unroutable messages land in a monitored queue instead of disappearing:

```python
# Declare the fallback exchange and an unroutable-catcher queue first.
channel.exchange_declare(exchange="orders.unroutable", exchange_type="fanout", durable=True)
channel.queue_declare(queue="unroutable_orders", durable=True)
channel.queue_bind(exchange="orders.unroutable", queue="unroutable_orders")

# Declare the main exchange with an alternate-exchange pointing at the fallback.
channel.exchange_declare(
    exchange="orders",
    exchange_type="topic",
    durable=True,
    arguments={"alternate-exchange": "orders.unroutable"},
)

# Now a publish with a typo'd key still lands somewhere observable.
channel.basic_publish(exchange="orders", routing_key="order.plced.eu", body=b"oops typo")
# -> no binding matches "order.plced.eu" -> routed to orders.unroutable -> unroutable_orders
```

### Binding arguments

Beyond the binding key, bindings can carry **arguments** — the `arguments` dict you saw on the headers exchange. For direct, fanout, and topic exchanges, binding arguments are mostly ignored (the routing key or its absence does the work). For headers exchanges, the arguments *are* the matching rule (`x-match` plus the header values). And plugin exchanges can define their own argument semantics — the consistent-hash exchange, for example, reads a weight from the binding. The general point: a binding is not just `(exchange, queue, key)`; it is `(exchange, queue, key, arguments)`, and which parts matter depends on the exchange type.

### Exchange-to-exchange bindings

Here is a feature most people never discover: you can bind an exchange to *another exchange*, not just an exchange to a queue. A binding's destination can be a queue (the usual case) or a second exchange, and when an exchange is the destination, the matched message is re-routed through *its* bindings in turn. This lets you build routing *topologies* — a tree or DAG of exchanges — rather than a flat one-hop routing table.

Why would you want this? The killer use case is layering a coarse routing stage in front of a fine one. Suppose every event in your system flows through a top-level `events` topic exchange keyed by `<domain>.<event>` — `orders.placed`, `payments.failed`, `users.created`. You bind a per-domain exchange to it: an `orders` exchange bound with `orders.#`, a `payments` exchange bound with `payments.#`. Each domain team then owns *their* exchange and wires their queues to it however they like, with their own routing-key conventions below the domain prefix, without touching the top-level exchange or coordinating with other teams. The top-level exchange is a stable, organization-wide contract; the per-domain exchanges are team-local. You have turned routing into a two-level namespace, and a new domain is a single new binding on the root.

```python
# Top-level fan-out by domain, then per-domain exchanges own their own routing.
channel.exchange_declare(exchange="events", exchange_type="topic", durable=True)
channel.exchange_declare(exchange="orders", exchange_type="topic", durable=True)

# Bind the 'orders' EXCHANGE (not a queue) to the root with a domain prefix.
channel.exchange_bind(destination="orders", source="events", routing_key="orders.#")

# A producer publishes once to the root; it cascades into the orders exchange,
# which then routes by its own bindings to the orders team's queues.
channel.basic_publish(exchange="events", routing_key="orders.placed.eu", body=b"...")
```

Use this sparingly — every extra hop is another routing evaluation and another place a message can fail to match — but for large multi-team event fabrics, exchange-to-exchange bindings keep the namespace clean and ownership local.

### Alternate exchanges in the publish path

The figure below shows where this fits in the lifecycle of a publish: the message arrives, the exchange matches bindings, matched messages are enqueued into queues and delivered to consumers, and unmatched messages take the side path to the alternate exchange instead of vanishing.

![A pipeline showing a publish flowing through binding match and enqueue to delivery, with an unroutable side path to an alternate exchange](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-9.webp)

## 9. A complete routing example end to end

Let us assemble everything into one realistic topology and walk a message all the way through, so the pieces lock together. We will build the routing layer for an order-processing system that has to serve four very different consumers off the same stream of order events:

1. **Fulfillment** needs every placed order, to ship it.
2. **EU analytics** needs only EU placed orders, for a regional dashboard.
3. **Fraud** needs only *flagged* orders, regardless of region, for review.
4. **Audit** needs *every* order event of any kind — placed, cancelled, refunded — for compliance, and must never miss one.

We choose a topic exchange named `orders` and a routing-key scheme `order.<action>.<region>.<flag>`, where `action` is `placed`/`cancelled`/`refunded`, `region` is `eu`/`us`/`apac`, and `flag` is `clean`/`flagged`. Example keys: `order.placed.eu.clean`, `order.placed.us.flagged`, `order.refunded.apac.clean`. We give the exchange an alternate exchange so a typo'd key cannot silently drop a compliance-relevant event.

Now the bindings, each expressing one consumer's interest:

| Queue | Binding key | Interest |
| --- | --- | --- |
| `fulfillment` | `order.placed.#` | all placed orders, any region, any flag |
| `eu_analytics` | `order.placed.eu.*` | placed EU orders, any flag |
| `fraud` | `order.*.*.flagged` | any action, any region, but flagged |
| `audit` | `order.#` | every order event of any shape |

Trace a publish of `order.placed.eu.flagged` — an EU order that was placed and flagged for review:

- `fulfillment` bound `order.placed.#` — literal `order`, literal `placed`, `#` matches `eu.flagged`. **Match.**
- `eu_analytics` bound `order.placed.eu.*` — `order`, `placed`, `eu` literal, `*` matches `flagged`. **Match.**
- `fraud` bound `order.*.*.flagged` — `order`, `*` matches `placed`, `*` matches `eu`, literal `flagged`. **Match.**
- `audit` bound `order.#` — `#` matches `placed.eu.flagged`. **Match.**

This one message lands in **all four** queues, each independently, with four independent backlogs and four independent acknowledgement streams. That is correct: this event is simultaneously a placed order (fulfillment), an EU placed order (analytics), a flagged order (fraud), and an order event (audit). The single routing key `order.placed.eu.flagged` encodes all four facts, and the four bindings each pick out the facet they care about.

Now trace `order.cancelled.us.clean` — a clean US cancellation:

- `fulfillment` `order.placed.#` — second word is `cancelled`, not `placed`. **No match.**
- `eu_analytics` `order.placed.eu.*` — `placed` mismatch. **No match.**
- `fraud` `order.*.*.flagged` — last word is `clean`, not `flagged`. **No match.**
- `audit` `order.#` — matches anything under `order`. **Match.**

So a clean cancellation reaches only `audit`. Exactly right — fulfillment does not ship cancellations, analytics tracks placements not cancellations, fraud only cares about flagged, and audit catches everything. The routing did all of that with zero consumer-side filtering and zero producer awareness.

Finally, trace a misconfiguration: a producer with a bug publishes `orders.placed.eu.clean` (note the typo — `orders` instead of `order`). No binding matches `orders.placed.eu.clean` because every binding key starts with the literal word `order`, not `orders`. Without an alternate exchange this compliance-relevant event would silently vanish. *With* the alternate exchange we declared, it routes to the fallback exchange and lands in an unroutable queue that an operator monitors — the event is preserved and the bug is caught, instead of becoming a silent gap in the audit log that surfaces during a compliance review months later. This is precisely why I treat alternate exchanges as mandatory for any stream where a dropped message is a real problem.

The full topology, declared once at deploy time:

```python
# One topic exchange with an alternate-exchange safety net.
channel.exchange_declare(exchange="orders.unroutable", exchange_type="fanout", durable=True)
channel.queue_declare(queue="unroutable_orders", durable=True)
channel.queue_bind(exchange="orders.unroutable", queue="unroutable_orders")

channel.exchange_declare(
    exchange="orders", exchange_type="topic", durable=True,
    arguments={"alternate-exchange": "orders.unroutable"},
)

for queue, key in [
    ("fulfillment",  "order.placed.#"),
    ("eu_analytics", "order.placed.eu.*"),
    ("fraud",        "order.*.*.flagged"),
    ("audit",        "order.#"),
]:
    channel.queue_declare(queue=queue, durable=True)
    channel.queue_bind(exchange="orders", queue=queue, routing_key=key)

# Producers publish a single key; bindings do the rest.
channel.basic_publish(exchange="orders", routing_key="order.placed.eu.flagged",
                      body=b'{"id": 42}')
```

This is the whole routing model in one screen: one exchange, four bindings, a safety net, and a producer that publishes one self-describing routing key without knowing or caring who consumes it. Everything in Part 2 — acks, confirms, durability, dead-lettering, quorum queues — is about making each of those four independent queues *reliable*; this post was about making sure the message reaches the right ones in the first place.

## The four exchange types side by side

Having seen each type in isolation, the comparison table earns its place. The figure below puts the four exchange types down the rows and three decision dimensions across the columns: how they read the routing key, how widely they fan out, and the typical use that follows.

![A matrix comparing direct, fanout, topic, and headers exchanges across routing rule, fan-out breadth, and typical use](/imgs/blogs/rabbitmq-amqp-exchanges-bindings-routing-5.webp)

And as a quick-reference, here is the same comparison as a table with the wildcard and matching semantics spelled out:

| Exchange | Reads | Matching rule | Fan-out | Reach for it when |
| --- | --- | --- | --- | --- |
| `direct` | routing key | exact string equality | narrow (queues with that key) | typed task queues, point-to-point dispatch |
| `fanout` | nothing | always matches | broad (all bound queues) | broadcast every event to every subscriber |
| `topic` | routing key | `*`=one word, `#`=zero-or-more | selective by pattern | multi-axis routing (region, severity, tenant) |
| `headers` | headers | `x-match` all/any on attributes | selective by attributes | non-hierarchical, multi-attribute routing |

The decision is almost always: exact value → direct; everyone → fanout; pattern over dotted axes → topic; non-string multi-attribute → headers (and you will rarely land on headers). Ninety percent of real topologies are topic exchanges with a well-designed routing-key scheme plus a few direct exchanges for simple task queues.

## Case studies and war stories

Theory is cheap. Here are four times the routing model bit hard or paid off in production, and the lesson each one teaches.

### Case study 1 — The silent routing typo that lost a day of webhooks

A payments team published webhook-delivery events to a topic exchange with routing key `webhook.send.<merchant>`. A deploy changed the key to `webhooks.send.<merchant>` — plural `webhooks` — in one of three publishing services. No binding matched the plural key. There was no alternate exchange, no `mandatory` flag. The broker accepted every publish and routed every one of them to *zero* queues. The producers saw success. Webhooks from that one service simply stopped being delivered, and because the metric the team watched was "publish errors" (which was zero), nothing fired. It took most of a day and a confused merchant to discover that a third of webhooks were vanishing into the routing void. **Lesson:** unroutable-by-default is a footgun. Always attach an alternate exchange (or `mandatory` with a return handler) to any exchange whose dropped messages would be a real problem. A routing-key typo should land in a lost-and-found queue with an alert, not in nothing.

### Case study 2 — Connection-per-request melting the broker

A team's API gateway opened a new AMQP connection on every inbound HTTP request to publish an audit event, then closed it. Under a load test at a few thousand requests per second, the RabbitMQ node's CPU pinned at 100%, almost entirely in connection setup and teardown and TLS handshakes — the *routing* work was trivial by comparison. Publish latency, which should have been sub-millisecond, climbed to hundreds of milliseconds because every publish was preceded by a full TLS handshake. The fix was a single shared, long-lived connection per gateway process with a small channel pool; broker CPU dropped from pinned to single digits and publish latency fell back under a millisecond. **Lesson:** connections are expensive and meant to be long-lived; channels are cheap and meant to be many. Connection-per-message is the most common RabbitMQ performance pathology, and the cure is structural — pool the connection, open channels per thread.

### Case study 3 — Fanout where a topic belonged

A notifications service used a fanout exchange to broadcast every user event to every downstream consumer, then had each consumer filter in code for the events it cared about. The email consumer wanted ~2% of events; it received 100% and threw away 98% after deserializing each one. As traffic grew, the email consumers spent most of their CPU deserializing and discarding messages they did not want, and their queues — which received every event — backed up under load that had nothing to do with email volume. Switching to a topic exchange with a routing-key scheme moved the filtering into the broker: the email queue bound only the patterns it wanted and received only ~2% of the traffic. Consumer CPU dropped sharply and the queue depth tracked actual email volume. **Lesson:** if your consumers filter the fanout in code, you wanted a topic exchange. Let the broker do the selection; that is what the routing model is *for*. Fanout is for "everyone genuinely wants everything," not "everyone gets everything and sorts it out."

### Case study 4 — One binding per tenant, and the topic exchange that crawled

A multi-tenant SaaS routed events to per-tenant queues using a topic exchange, with the tenant ID baked into the routing key as `tenant.<uuid>.<event>` and *one exact binding per tenant queue* — `tenant.<uuid>.#`. At a few hundred tenants this was fine. As the product grew past tens of thousands of tenants, the topic exchange held tens of thousands of bindings, and the per-message routing cost — walking the binding trie for every publish — became a measurable fraction of broker CPU at peak. Worse, declaring and tearing down bindings as tenants signed up and churned generated a steady stream of topology changes that the broker had to propagate. The team had turned a topic exchange into a hash table with tens of thousands of keys, which is not what the matching engine is built for. The fix was to stop using one binding per tenant and instead route with the consistent-hash exchange plugin, which hashes the routing key to pick one of a small fixed set of queues, giving tenant-stable partitioning with a handful of bindings instead of one per tenant — the pattern the [production scaling post](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) details. **Lesson:** topic matching cost scales with binding count. Wildcards over a small set of axes are cheap; tens of thousands of exact bindings are not. When your routing axis is high-cardinality identity, hash it; do not enumerate it as bindings.

### Case study 5 — Channel shared across threads, corrupting the stream

A Java service using a thread pool shared one `Channel` object across all worker threads to "save resources." Under concurrency, two threads occasionally published on the same channel simultaneously, interleaving AMQP frames on the wire. The result was intermittent `UNEXPECTED_FRAME` connection errors that killed the *entire connection* (and thus every channel on it), roughly once an hour under load, with no clear pattern. Days were lost chasing a "network glitch" that was actually a thread-safety violation. The fix was one channel per thread (a small thread-local channel pool). **Lesson:** a channel is not thread-safe; a connection is. Share the connection, give each thread its own channel. This is documented and routinely ignored, and the symptom — rare, timing-dependent connection drops — points nowhere near the actual cause.

## When to reach for this (and when not to)

The routing model is RabbitMQ's defining strength, and it is the right tool when *routing complexity* is your problem.

**Reach for RabbitMQ's exchange routing when:**

- You need **flexible, evolving fan-out** where consumers come and go and the producer must not change. Topic exchanges plus consumer-declared bindings are purpose-built for this.
- Your routing depends on **multiple axes** (region, severity, tenant, event type) that you want to slice independently. A topic routing-key scheme encodes all of them in one string and lets each consumer slice as it likes.
- You want **broker-side selection** so consumers receive only what they asked for, not everything-and-filter. This keeps consumers simple and CPU-efficient.
- You need **per-message routing decisions** — this message to fraud, that one not — driven by message content expressed in the key. The broker routes; the consumer just processes.
- You want **point-to-point work queues with typed dispatch** — a direct exchange with one binding key per worker pool is clean and fast.

**Do not reach for it (reach for a log like Kafka instead) when:**

- You need **replay** — "reprocess the last 30 days." RabbitMQ deletes on ack; the routing model has no offsets, no rewind. This is covered in the [queue vs pub/sub vs log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) trade-off and is the single biggest reason teams outgrow RabbitMQ.
- You need **very high per-partition throughput** with strict total order — millions of messages per second through one ordered stream. RabbitMQ's per-queue throughput tops out far below that; the routing flexibility costs you raw streaming speed.
- Your data is a **durable event log** that many independent consumer groups read at different speeds from arbitrary historical positions. That is Kafka's model, not RabbitMQ's; faking it with shadow queues is a system you will hate.
- You need **massive numbers of high-cardinality exact routes** (one binding per of hundreds of thousands of keys). Topic matching cost grows with bindings; for partition-style routing at scale, the consistent-hash exchange or a different broker fits better — see the [production scaling post](/blog/software-development/system-design/rabbitmq-production-architecture-scaling).

The honest summary: RabbitMQ's routing model is unmatched for *expressive, evolving, content-based routing* of discrete messages that get processed and deleted. It is the wrong tool when you actually want a replayable, high-throughput, append-only log. Most systems that "outgrow RabbitMQ" did not outgrow its routing; they discovered they needed a log all along.

## Key takeaways

- **Producers publish to exchanges, never to queues.** The exchange is a stateless routing function; bindings decide which queues receive the message. This indirection is the whole point — it decouples producer code from consumer topology.
- **The default (nameless) exchange makes `basic_publish("", "queue")` look like publishing to a queue,** but it is really a direct exchange with an automatic binding per queue. Use named exchanges in production so routing is explicit.
- **Reuse a few long-lived connections; open many cheap channels.** A connection is a TCP/TLS socket and is expensive; a channel multiplexes over it and is cheap. Connection-per-message is the most common performance pathology, and channels are not thread-safe — one channel per thread.
- **Four exchange types, four routing functions:** direct (exact key match), fanout (broadcast, key ignored), topic (`*` matches one word, `#` matches zero-or-more), headers (match on attributes with `x-match` all/any). Topic plus a good routing-key scheme covers most real needs.
- **Design the routing key as your API:** order the dotted axes by how you slice them, keep the arity stable per event family, and consumers compose bindings against it without producer changes.
- **Unroutable messages are silently dropped by default.** Attach an alternate exchange (preferred, broker-side) or publish with `mandatory` and a return handler so a routing-key typo lands in a monitored lost-and-found rather than vanishing.
- **Routing key (producer-set) vs binding key (consumer-set):** the exchange compares them per its type's rule. Keep the three roles — routing key, binding key, exchange comparison — straight and the model reads cleanly.
- **A single self-describing routing key can fan out to exactly the right set of consumers** with zero consumer-side filtering, as the end-to-end order example shows. That is the routing model working as designed.

## Further reading

- [RabbitMQ Deep Dive, Part 2: acks, publisher confirms, durability, and quorum queues](/blog/software-development/message-queue/rabbitmq-acks-confirms-durability-quorum-queues) — the reliability half of this story: how a message in flight is guaranteed and how the queue survives a crash.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — clustering, federation, quorum queues, the consistent-hash exchange, and operational scaling patterns that pick up where this routing model leaves off.
- [Queue vs Pub/Sub vs Log: three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) — the conceptual map that explains *why* RabbitMQ routes and Kafka logs, and how RabbitMQ builds all three shapes from these primitives.
- [Push vs pull and acknowledgements: how consumers read](/blog/software-development/message-queue/push-vs-pull-acknowledgements-how-consumers-read) — how messages get from a queue into a consumer, and how acks close the loop on the routing this post sets up.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the contrasting model where storage is the core idea and routing is just a partition number plus an offset.
- The official AMQP 0-9-1 model and reference, plus the RabbitMQ tutorials on exchanges, bindings, and topic routing — the authoritative source for the protocol semantics summarized here.
