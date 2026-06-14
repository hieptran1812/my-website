---
title: "Choosing a Message Broker: Kafka vs RabbitMQ vs Pulsar vs NATS vs SQS"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A decision framework for picking a message broker: profile Kafka, RabbitMQ, Pulsar, NATS, and SQS by their real sweet spots and failure modes across throughput, routing, replay, ordering, ops, and cost, with a decision tree, a positioning map, and the anti-patterns that sink projects."
tags:
  [
    "message-queue",
    "kafka",
    "rabbitmq",
    "pulsar",
    "nats",
    "sqs",
    "distributed-systems",
    "event-driven",
    "system-design",
    "broker-selection",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-1.webp"
---

The worst broker decisions I have watched get made were not made by people who knew too little. They were made by people who knew one broker very well and reached for it reflexively for everything. The Kafka shop that put a synchronous request/reply RPC on a partitioned log and then spent a quarter fighting tail latency. The RabbitMQ shop that needed to replay six months of events for a new analytics consumer and discovered, far too late, that every message had been deleted the instant it was acknowledged. The startup that picked SQS because "it's zero-ops" and then needed strict global ordering for a financial ledger, which SQS standard queues simply do not provide. None of these teams were incompetent. They were pattern-matching on familiarity instead of on the shape of the problem.

This post is a decision framework, not a feature listicle. I am not going to recite each broker's spec sheet and let you draw your own conclusions, because the spec sheets all look impressive and none of them tell you where the broker *breaks*. Instead I am going to profile each of the five brokers that dominate real deployments — Kafka, RabbitMQ, Pulsar, NATS/JetStream, and Amazon SQS/SNS — and for each one I will tell you its honest sweet spot *and* its honest failure mode. A tool you cannot describe the failure mode of is a tool you do not understand. The figure below is the entire argument compressed into one table: the five brokers across the top, and the five decision axes that actually move the needle down the side.

![A comparison matrix showing five brokers across the columns and the axes of throughput, routing, replay, operational effort, and latency down the rows, with each broker strong on different axes](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-1.webp)

By the end of this post you will be able to walk into a design review, hear a messaging requirement described, and route it to the right broker in under a minute — with a defensible reason, not a vibe. You will have a decision tree you can apply mechanically, a positioning map that shows you why high throughput and rich routing pull in opposite directions, and a catalog of the anti-patterns that turn a good broker into a production incident. We will close by mixing brokers in a single architecture, because the mature answer to "which broker?" is frequently "more than one, with a clear boundary between them."

This is post twelve in a forty-part message-queue series. It builds directly on the conceptual map laid out in [Queue vs Pub/Sub vs Log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models), and it leans on the broker-specific deep dives elsewhere in the series: [Kafka storage internals](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage), [Kafka replication and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability), and [RabbitMQ AMQP exchanges and routing](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing). Where those posts go deep on one machine, this one stands back and draws the whole territory so you know which machine to pick up.

## 1. The decision axes that actually matter

Before we touch a single broker, we have to agree on the axes we are scoring them against. The most common failure in broker selection is choosing the wrong axes — benchmarking raw throughput when your real constraint is operational headcount, or obsessing over feature richness when your real constraint is tail latency. There are eight axes that, in my experience operating these systems, capture nearly every consequential difference. Get these straight and the brokers practically sort themselves.

The figure at the top of the post scores the five brokers across the five most decisive of these axes. Let me define all eight precisely, because vague definitions are how people talk past each other in design reviews.

**Throughput** is sustained messages per second (and the related metric, bytes per second) that a single cluster can absorb and deliver without falling behind. This is the axis everyone benchmarks and the one that matters *least* often, because most workloads are nowhere near any broker's ceiling. A broker that does 50,000 messages per second is plenty for the overwhelming majority of business applications. Throughput only becomes the deciding axis when you are genuinely in the hundreds-of-thousands-to-millions range: clickstream, telemetry, IoT, log aggregation, market data.

**Latency** is the time from publish to delivery, and you must always ask *which percentile*. A broker with a 2-millisecond median and a 2-second p99 is a completely different animal from one with a 10-millisecond median and a 15-millisecond p99. For task distribution and request/reply, tail latency dominates user experience. For batch analytics, latency barely matters at all. Brokers that batch aggressively for throughput (Kafka) pay for it in latency; brokers tuned for low latency (RabbitMQ, NATS) give up some peak throughput.

**Routing complexity** is how much logic the broker can apply to decide *which consumers* see a given message. At the simple end, a message goes onto a named queue and whoever reads that queue gets it. At the rich end, the broker inspects message attributes and a topology of exchanges and bindings to route one publish to a specific subset of queues by pattern matching. RabbitMQ's AMQP model is the gold standard here; Kafka deliberately offers almost none of this, pushing routing into partition keys and consumer-side filtering.

**Replay and retention** is whether a consumer can re-read messages it already processed, and how far back history goes. A log keeps messages after they are read and lets any reader rewind to any offset within the retention window. A queue deletes on acknowledgement, so there is nothing to replay. This single axis is the sharpest dividing line in the whole field. If you need to onboard a new consumer that processes all of last month's events, only a log-shaped broker can do it.

**Ordering guarantees** is what order consumers observe messages in, and across what scope. The options range from "no ordering at all" through "ordering within a partition or key" to "strict total order across the entire topic" (which essentially nobody offers at scale because it kills parallelism). Most real systems want *per-entity* ordering — all events for a given customer or order in sequence — which is exactly what partition-key ordering provides.

**Operational burden** is what it costs you in human attention to run the thing: how many components, how hard upgrades are, how it behaves under partial failure, how much expertise the on-call rotation needs. This axis is routinely undervalued because it does not show up in a benchmark, and then it dominates the total cost of ownership. A broker that needs a dedicated platform team is a profoundly different commitment from a single binary one engineer can operate on the side.

**Ecosystem and tooling** is the depth of client libraries, connectors, stream-processing frameworks, monitoring integrations, managed offerings, and the Stack Overflow corpus you can lean on at 3 a.m. Kafka's ecosystem is enormous — Kafka Connect, Kafka Streams, ksqlDB, Schema Registry, a hundred sink and source connectors. A broker with a thin ecosystem makes you build the glue yourself.

**Cost** is the all-in price: infrastructure, licensing or managed-service fees, *and* the engineering time the operational burden axis already hinted at. Self-hosted brokers shift cost from per-message fees to fixed infrastructure plus salaried operations. Managed brokers like SQS charge per request, which is cheap at low volume and can become eye-watering at high volume. Cost interacts with every other axis — you are usually paying to relieve operational burden or to gain throughput headroom.

### Why no single broker wins

Here is the load-bearing observation: these axes are not independent, and several of them are in genuine *tension*. Rich routing requires the broker to track per-message state and make per-message decisions, which caps throughput. High throughput requires append-only batched writes and dumb fan-out, which precludes rich routing. Replay requires retaining messages on disk, which a delete-on-ack queue cannot do without becoming a log. Low operational burden usually means either a single simple binary (giving up some scale features) or a managed service (giving up control and paying per message). You cannot maximize all eight axes simultaneously because the underlying mechanisms conflict.

That is the whole reason this post exists. If one broker dominated every axis, the decision would be trivial and nobody would argue about it. Instead, each broker is a *point in this eight-dimensional tradeoff space*, having chosen which axes to optimize and which to sacrifice. Choosing a broker is choosing which sacrifices you can live with. So let us profile each one by exactly that: what it optimizes, and what it gives up.

## 2. Kafka: the streaming log

Kafka is a distributed, partitioned, replicated commit log. That is not marketing; it is the literal data structure. A topic is split into partitions, each partition is an append-only file (a sequence of [log segments backed by the OS page cache](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage)), and producers append to the tail while consumers read forward at their own pace by tracking an offset. Messages are retained for a configured time or size regardless of whether anyone has read them, and replication across [an in-sync replica set governs durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability). This architecture is the source of every Kafka strength and every Kafka weakness, which is why understanding it pays off so well.

### What Kafka is great at

Kafka's sweet spot is **high-throughput event streaming with replay**. Because writes are sequential appends batched into large I/O operations and served to consumers via zero-copy `sendfile`, a modest cluster sustains hundreds of thousands to millions of messages per second on commodity hardware. Because messages are retained, you get **replay for free**: a new consumer group can start from offset zero and reprocess all of history, a stream-processing job can be restarted from a checkpoint and recompute, and a buggy consumer can be fixed and rewound to re-handle the messages it mangled. This replay property is genuinely hard to replicate in queue-shaped brokers and is the single biggest reason teams reach for Kafka.

Kafka also anchors a vast stream-processing ecosystem. Kafka Streams and ksqlDB let you do stateful aggregations, joins, and windowing directly on topics. Kafka Connect moves data in and out of dozens of systems without custom code. This makes Kafka not just a broker but the backbone of a data platform — which is exactly how the largest deployments use it, as the central nervous system that every service publishes to and every analytical system reads from.

```python
# Kafka producer tuned for throughput: large batches, compression, idempotence.
from confluent_kafka import Producer

producer = Producer({
    "bootstrap.servers": "kafka-1:9092,kafka-2:9092,kafka-3:9092",
    "acks": "all",                  # wait for all in-sync replicas: durability
    "enable.idempotence": True,     # no duplicates on producer retry
    "compression.type": "lz4",      # cheap CPU for big network/disk savings
    "linger.ms": 20,                # wait up to 20ms to fill a batch
    "batch.size": 262144,           # 256KB batches amortize per-message cost
})

def on_delivery(err, msg):
    if err is not None:
        # Surface and alert; do not silently drop.
        print(f"delivery failed: {err}")

# Key drives the partition, which drives ordering for this entity.
producer.produce("orders", key=order_id, value=payload, callback=on_delivery)
producer.flush()
```

That `linger.ms=20` line is the throughput-versus-latency tradeoff made explicit: you are telling Kafka to wait up to twenty milliseconds to accumulate a fat batch. Great for throughput, a tax on latency. This knob is a microcosm of Kafka's whole personality.

### Where Kafka breaks

Kafka's failure mode is **anything that needs per-message routing or low-latency request/reply**. Kafka has essentially no routing engine. A message goes to a partition determined by its key's hash, and that is the entire routing story. You cannot say "deliver this message only to consumers interested in attribute X" at the broker; you either create separate topics or you filter on the consumer side after reading everything. Teams that need content-based routing on Kafka end up building a fragile mess of topics or a custom routing service, reinventing what RabbitMQ gives them out of the box.

Kafka is also a poor fit for **synchronous request/reply RPC**. The consumer model is a poll loop, which adds latency, and getting a response back requires a reply topic plus correlation IDs plus consumer-side matching — a lot of machinery to simulate something a queue or an RPC framework does natively. Worse, head-of-line blocking within a partition means one slow message can stall everything behind it; a single misbehaving consumer can spike p99 latency for an entire partition.

And Kafka is **operationally heavy**. Even after KRaft removed the ZooKeeper dependency, you are running a stateful distributed system with partition leadership, replication, rebalancing, and retention to manage. Partition count is a sizing decision you live with for years. Rebalances can pause consumption. Capacity planning, broker upgrades, and disaster recovery all need real expertise. Kafka rewards a platform team and punishes a team that wanted to "just send some messages."

| Property | Kafka's answer |
| --- | --- |
| Model | Partitioned append-only log |
| Throughput | Very high (100k–1M+ msg/s) |
| Latency | Milliseconds; tunable but batch-biased |
| Routing | Key-to-partition only; no content routing |
| Replay | Yes — core feature, retained by time/size |
| Ordering | Per-partition (per-key) total order |
| Ops burden | High — stateful cluster, partitions, rebalances |
| Best for | Event streaming, log aggregation, stream processing, CDC |

## 3. RabbitMQ: the routing queue

RabbitMQ is the most capable *router* in this comparison, and it comes by that honestly: it implements AMQP 0-9-1, a protocol designed around the idea that a publisher should not know or care which queues its message ends up in. Publishers send to an **exchange**, and bindings between exchanges and queues — evaluated against the message's routing key — decide where the message lands. This indirection is the heart of RabbitMQ's power and is covered in depth in the [RabbitMQ AMQP exchanges and routing](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing) post in this series. Here I care about *when to pick it*.

### What RabbitMQ is great at

RabbitMQ's sweet spot is **low-latency task distribution with complex routing and fine-grained per-message control**. If you have a pool of workers chewing through jobs and you want each job done once by whichever worker is free, RabbitMQ's competing-consumers queue is the canonical, battle-tested answer, with sub-millisecond broker latency. Layer on the routing engine and you can express genuinely sophisticated topologies: a direct exchange for exact-match routing, a topic exchange for pattern-based fan-out (`logs.*.error` to one queue, `logs.payment.#` to another), a fanout exchange for broadcast, and a headers exchange for routing on arbitrary attributes. One publish can fan out to exactly the right subset of queues without the publisher knowing anything about the consumers.

RabbitMQ also gives you **per-message control** that log-based brokers cannot. Per-message TTL, priority queues, dead-letter exchanges for messages that fail too many times, publisher confirms, consumer acknowledgements with explicit requeue or reject, and per-message expiration. You can express "retry this three times, then send it to the failures queue, and if it sits unprocessed for an hour, expire it" declaratively. This granular control is exactly what task-processing systems need and exactly what a log does not offer, because a log only knows offsets, not per-message lifecycle.

```python
# RabbitMQ: a topic exchange routing orders to region-specific queues,
# with a dead-letter exchange for poison messages.
import pika

conn = pika.BlockingConnection(pika.ConnectionParameters("rabbit-1"))
ch = conn.channel()

ch.exchange_declare(exchange="orders", exchange_type="topic", durable=True)
# Dead-letter exchange catches messages rejected too many times.
ch.exchange_declare(exchange="orders.dlx", exchange_type="fanout", durable=True)

ch.queue_declare(queue="orders.us", durable=True, arguments={
    "x-dead-letter-exchange": "orders.dlx",
    "x-max-priority": 10,           # priority queue: urgent orders jump ahead
})
ch.queue_bind(queue="orders.us", exchange="orders", routing_key="orders.us.*")

# Publisher confirms make the broker ack the publish for durability.
ch.confirm_delivery()
ch.basic_publish(
    exchange="orders",
    routing_key="orders.us.express",
    body=payload,
    properties=pika.BasicProperties(delivery_mode=2, priority=9),  # persistent
)
```

### Where RabbitMQ breaks

RabbitMQ's failure mode is **replayable history and very high throughput**. The defining property of a queue is consume-and-delete: once a consumer acknowledges a message, it is gone. There is no offset to rewind, no retained history to re-read. If a new analytics team shows up six months in and asks to reprocess all past orders, RabbitMQ has nothing for them — those messages were deleted the moment they were acknowledged. Streams (RabbitMQ 3.9+) add a log-like retained structure that softens this, but it is bolted onto a system whose heart is the delete-on-ack queue, and it does not match Kafka's replay ergonomics or throughput.

RabbitMQ's throughput ceiling is also materially lower than the log brokers'. A single queue is effectively single-threaded for ordering purposes, and the rich per-message bookkeeping that makes routing powerful costs CPU and memory per message. Real deployments do tens of thousands of messages per second per queue comfortably and can be scaled with more queues and sharding, but you will not casually hit the million-messages-per-second figures that Kafka reaches with sequential appends. Deep queue backlogs are a known operational hazard: when a queue grows beyond what fits in memory, RabbitMQ pages messages to disk and latency degrades, and a flooded broker under memory pressure can stop accepting publishes entirely. RabbitMQ wants its queues *short* — it is a flow-through router, not a long-term store.

| Property | RabbitMQ's answer |
| --- | --- |
| Model | AMQP queue with exchange routing |
| Throughput | Moderate (tens of thousands/s per queue) |
| Latency | Sub-millisecond broker latency |
| Routing | Best in class — exchanges, bindings, headers |
| Replay | No (Streams add limited retention) |
| Ordering | Per-queue FIFO (with caveats under redelivery) |
| Ops burden | Moderate — clustering, mirroring, memory watermarks |
| Best for | Task queues, RPC-style work, complex routing |

## 4. Pulsar: the hybrid with tiered storage

Apache Pulsar is the most architecturally interesting broker in this group, and the one whose tradeoffs are the least understood. Its defining choice is **separating serving from storage**. Brokers in Pulsar are stateless — they own no data on local disk. The actual messages live in Apache BookKeeper, a separate distributed log-storage system whose nodes are called bookies, which write to write-ahead ledgers. Cold data can then be offloaded to tiered object storage like S3. The figure below shows this layering, and it is the key to understanding both why Pulsar is powerful and why it has more moving parts than anything else here.

![A layered stack diagram showing producers and consumers on top, then stateless Pulsar brokers, then BookKeeper bookies for storage, then tiered object storage, with a metadata store alongside](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-6.webp)

### What Pulsar is great at

The separation of compute and storage is not academic; it buys real operational properties. Because brokers are stateless, you can add or remove them in seconds without rebalancing data — there is no data on them to move. Scaling serving capacity is decoupled from scaling storage capacity, so a workload that is throughput-bound and a workload that is retention-bound scale independently. This is genuinely hard to do in Kafka, where a broker owns its partitions' data and adding a broker means a slow, I/O-heavy reassignment.

Pulsar's second sweet spot is **unifying the log and the queue in one system**. A Pulsar topic can be consumed in *exclusive* or *failover* mode (log-like, one active consumer per partition, ordered) or in *shared* mode (queue-like, competing consumers, each message to one consumer) or *key-shared* mode (per-key ordering with multiple consumers). The same topic, the same retained data, two different consumption contracts. This is the consider-the-case scenario where you genuinely want both a replayable log *and* competing-consumer work distribution over the same stream without running two separate systems.

Third, Pulsar was built for **native multi-tenancy and geo-replication**. Tenants, namespaces, and per-namespace policies are first-class, so a single Pulsar cluster can safely host many teams with isolation and quotas. Geo-replication across regions is configured declaratively rather than bolted on. For a platform team serving the whole company across multiple data centers, these are exactly the properties you want and exactly the ones that are painful to assemble on Kafka.

### Where Pulsar breaks

Pulsar's failure mode is **operational complexity and a thinner ecosystem**. Look at that stack again: you are running brokers, BookKeeper bookies, a metadata store (ZooKeeper or etcd), and optionally tiered storage and the proxy. That is more components than any other broker here, and each is a thing to monitor, tune, upgrade, and reason about during an incident. BookKeeper in particular has its own performance characteristics and failure modes that your team must learn. The promise of independent scaling is real, but you pay for it with a larger surface area. For a small team, this complexity can outweigh the architectural elegance.

The ecosystem, while growing, is meaningfully behind Kafka's. There are fewer connectors, less third-party tooling, a smaller pool of engineers who have operated it, and a thinner corpus of production war stories to learn from. Pulsar offers a Kafka-compatibility layer to ease migration, but running a compatibility shim is its own complication. Pulsar is an excellent choice when its specific strengths — independent scale, unified queue/log semantics, native multi-tenancy and geo-replication — are things you concretely need. It is an over-investment when you do not, because you pay the complexity cost regardless of whether you use the features that justify it.

There is a subtler operational trap worth naming. Because Pulsar's storage lives in BookKeeper rather than on the broker, a Pulsar incident can originate in a layer most engineers are less familiar with than Kafka's single-process model. A slow bookie, an under-provisioned write-ahead-log disk, or a ledger that cannot reach its replication quorum manifests as broker-side latency or unavailability, and diagnosing it requires understanding the BookKeeper layer, the broker layer, and how they interact. Teams that adopt Pulsar for the elegant compute/storage split sometimes discover that the split also splits their *debugging surface* into two systems that fail in correlated but non-obvious ways. None of this is a reason to avoid Pulsar — it is a reason to budget real learning time before you depend on it, and to staff the adoption with people who will own that learning. The architectural elegance is real; so is the cost of the components that deliver it.

| Property | Pulsar's answer |
| --- | --- |
| Model | Segment-based log + queue hybrid |
| Throughput | Very high (comparable to Kafka) |
| Latency | Low milliseconds |
| Routing | Subscription modes; less than AMQP |
| Replay | Yes — retained, with tiered offload to S3 |
| Ordering | Per-partition / per-key (key-shared mode) |
| Ops burden | Highest — broker, bookie, metadata, tiered store |
| Best for | Multi-tenant platforms, geo-replication, unified queue+log |

## 5. NATS / JetStream: lightweight and fast

NATS is the broker that wins the axis everyone forgets to score: *operational simplicity*. The core NATS server is a single statically-linked binary with no external dependencies. You download it, run it, and you have a messaging system doing millions of messages per second at sub-millisecond latency. There is no ZooKeeper, no separate storage tier, no JVM tuning. For edge deployments, microservice meshes, and teams that want messaging without a platform team, this simplicity is not a minor nicety — it is the entire value proposition.

The figure below shows the two faces of NATS. Core NATS is fire-and-forget pub/sub: a publish goes to a subject, and every subscriber currently listening on that subject gets it, at most once, with no persistence. JetStream is the persistence layer added on top: it captures messages published to a subject into a durable stream that supports replay, durable consumers, and acknowledgements — log-like semantics over the same lightweight server.

![A graph showing a publisher sending to a NATS server, which fans out to two live subscribers and also persists into a JetStream durable store that feeds a pull consumer for replay](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-7.webp)

### What NATS is great at

NATS's sweet spot is **fast, simple messaging for microservices and edge**. The subject-based addressing model is elegant: subjects are dot-delimited hierarchies (`orders.us.express`) and subscriptions can use wildcards (`orders.*.express` or `orders.>`), giving you a lightweight form of topic routing without RabbitMQ's exchange machinery. Core NATS supports request/reply natively — a built-in pattern where a publisher gets a response on an automatically-managed inbox subject — which makes it genuinely good at the synchronous RPC use case that Kafka is bad at. Service meshes and microservice fleets use NATS as a fast internal bus precisely because it is low-latency, supports request/reply, and costs almost nothing to operate.

JetStream extends this to **persistence and replay** when you need it. You can define a stream that retains messages by count, size, or age, attach durable pull or push consumers with explicit acknowledgement, and replay from any point. It gives you log-like durability without leaving the NATS operational model — the same single binary, clustered for high availability with the Raft consensus protocol. For a team that wants *mostly* lightweight messaging but *occasionally* needs durability and replay, JetStream covers both without forcing a second system into the architecture.

```bash
# NATS is genuinely one binary. Start a JetStream-enabled server:
nats-server --jetstream --store_dir /data/nats

# Create a durable stream retaining 7 days of order events:
nats stream add ORDERS \
  --subjects "orders.>" \
  --storage file \
  --retention limits \
  --max-age 168h \
  --replicas 3

# A durable pull consumer that can replay from the start:
nats consumer add ORDERS billing \
  --pull --deliver all --ack explicit
```

### Where NATS breaks

NATS's failure mode is its **smaller ecosystem and feature surface**. JetStream is younger and less battle-hardened than Kafka's storage layer, and at extreme retention and throughput it does not yet have the decade of production scar tissue that Kafka has accumulated. The connector ecosystem is thin compared to Kafka Connect — if you need to sink data into twenty different downstream systems, you will write more glue yourself. Stream processing on NATS is nascent; there is no mature equivalent of Kafka Streams or Flink integration of the same depth. And core NATS's at-most-once delivery means that without JetStream, a subscriber that is down when a message is published simply misses it — fine for telemetry and presence, unacceptable for anything that must not be lost.

The honest summary: NATS is the right call when you value low latency and trivial operations and your durability needs are modest or well-served by JetStream. It is the wrong call when you need a deep stream-processing ecosystem, an enormous library of connectors, or the specific reassurance that comes from running the most-deployed log broker on the planet. NATS trades ecosystem depth for operational lightness, and that trade is excellent for many teams and wrong for some.

| Property | NATS / JetStream's answer |
| --- | --- |
| Model | Subject pub/sub + JetStream durable streams |
| Throughput | Very high (millions/s core) |
| Latency | Sub-millisecond |
| Routing | Subject wildcards; lighter than AMQP |
| Replay | Yes with JetStream; no with core NATS |
| Ordering | Per-subject / per-stream sequence |
| Ops burden | Lowest of the self-hosted — one binary |
| Best for | Microservices, edge, request/reply, IoT |

## 6. SQS / SNS: managed and zero-ops

Amazon SQS (Simple Queue Service) and SNS (Simple Notification Service) are the odd ones out in this comparison, because their defining property is not an architecture — it is the *absence of an architecture you have to run*. SQS is a fully managed queue; SNS is a fully managed pub/sub topic. You make API calls, AWS runs everything, and you never think about brokers, partitions, replication, disk, or upgrades. There is no cluster to size, no node to patch, no rebalance to dread. For an enormous number of teams, that is the single most valuable property a broker can have.

### What SQS/SNS is great at

The sweet spot is **zero operational burden and effectively infinite scale**. SQS standard queues scale to nearly unlimited throughput automatically — you do not provision capacity; you just send and receive, and AWS absorbs the load. There is no broker to fall over at 2 a.m., no capacity planning meeting, no on-call expertise to build. SNS provides fan-out: publish once to a topic, and it delivers to many subscribers — SQS queues, Lambda functions, HTTP endpoints, email. The canonical AWS pattern is SNS-to-SQS fan-out: one event published to SNS lands in several SQS queues, each owned by a different consumer, combining broadcast with durable per-consumer buffering. It is the path of least resistance for event-driven architectures inside AWS, and the integration with the rest of AWS (Lambda triggers, EventBridge, IAM) is seamless.

SQS also handles the un-glamorous reliability work for you. At-least-once delivery, a visibility timeout that hides an in-flight message from other consumers until it is processed or times back out, automatic retries, and a built-in dead-letter queue for messages that exceed a max receive count. These are exactly the primitives a task-processing system needs, and you get them without building or operating anything.

```python
# SQS: receive with long polling and a visibility timeout, then delete on success.
import boto3

sqs = boto3.client("sqs")
QUEUE = "https://sqs.us-east-1.amazonaws.com/123456789012/orders"

resp = sqs.receive_message(
    QueueUrl=QUEUE,
    MaxNumberOfMessages=10,
    WaitTimeSeconds=20,          # long polling: fewer empty calls, lower cost
    VisibilityTimeout=60,        # message hidden from others for 60s while we work
)
for msg in resp.get("Messages", []):
    try:
        handle(msg["Body"])
        # Delete only after successful processing -> at-least-once semantics.
        sqs.delete_message(QueueUrl=QUEUE, ReceiptHandle=msg["ReceiptHandle"])
    except Exception:
        # Do not delete: message reappears after VisibilityTimeout for retry.
        # After maxReceiveCount, it lands in the dead-letter queue automatically.
        pass
```

### Where SQS/SNS breaks

SQS's failure modes are sharp and worth memorizing. First, **ordering**. Standard SQS queues provide *no* ordering guarantee — messages can and do arrive out of order. SQS FIFO queues restore strict ordering and exactly-once processing, but they cap throughput (a few thousand messages per second per message group, higher with high-throughput mode) and cost more. So if your workload needs strict ordering at high volume, SQS is a poor fit — you are forced into FIFO and into its throughput limits. This is the anti-pattern that catches the most teams: reaching for cheap, infinite standard SQS for a workload that quietly assumed ordering.

Second, **no replay and limited retention**. SQS is a queue: consume-and-delete. There is no offset to rewind, and the maximum retention is 14 days, after which messages vanish whether consumed or not. You cannot onboard a new consumer to reprocess history, because there is no history to reprocess. If you need a replayable event log, SQS is categorically wrong; you want Kafka, Pulsar, or Kinesis (which is AWS's log-shaped offering and a different service entirely).

Third, **per-message cost and lock-in**. SQS charges per request (with batching to amortize), which is cheap at low volume and can become a large bill at sustained high volume — the cost model that is a bargain at 100 messages per second can be a budget line item at 50,000. And it is AWS-only: building on SQS/SNS couples your messaging layer to AWS, which matters if multi-cloud or portability is a real requirement. We will put concrete numbers on the cost crossover in the worked example below.

| Property | SQS / SNS's answer |
| --- | --- |
| Model | Managed queue (SQS) + managed pub/sub (SNS) |
| Throughput | Effectively unlimited (standard); capped (FIFO) |
| Latency | Tens of milliseconds typical |
| Routing | SNS topic fan-out; SNS filter policies |
| Replay | No; max 14-day retention |
| Ordering | None (standard); strict (FIFO, throughput-capped) |
| Ops burden | Zero — fully managed |
| Best for | AWS-native apps, zero-ops task queues, fan-out |

## 7. A decision tree for picking one

Now we assemble the framework. You have five honest profiles; the question is how to route a requirement to one of them quickly. I use a decision tree that asks the highest-leverage questions first — the ones that eliminate the most candidates per answer. The figure below is that tree, and I apply it in real design reviews.

![A decision tree that asks whether you need replay, then whether you need complex routing, then whether you want a managed service, routing each answer to a specific broker](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-2.webp)

The first question is the sharpest dividing line in the field: **do you need replay or retention?** Can a new consumer arrive later and need to reprocess history? Will you restart a stream-processing job from a checkpoint and recompute? Do you want events kept as a durable record independent of consumption? If yes, you are in log territory: **Kafka** if you want the deepest ecosystem and the standard choice, **Pulsar** if you additionally need independent compute/storage scaling, native multi-tenancy, or geo-replication. This question alone eliminates the three queue-shaped brokers, because none of them retains history for replay.

If you do *not* need replay, the second question is **do you need complex per-message routing?** Do you route messages to different consumers based on content, attributes, or patterns? Do you need per-message TTL, priorities, dead-letter handling, and fine-grained acknowledgement control? If yes, **RabbitMQ** is the answer — its AMQP routing is unmatched, and the per-message control is exactly what sophisticated task systems need. Nothing else in this list routes as expressively.

If you need neither replay nor rich routing — you just want to move messages between services reliably — the third question is operational: **do you want a fully managed, zero-ops service?** If yes and you are on AWS, **SQS/SNS** is the path of least resistance: no cluster, infinite scale, pay per use. If you would rather self-host something fast and trivially simple — perhaps for latency, cost at scale, multi-cloud portability, or edge deployment — then **NATS/JetStream** is the lightweight champion, one binary doing millions of messages per second with optional durability.

That is the whole tree, and it resolves most real cases in three questions. The order matters: replay is asked first because it is the hardest requirement to bolt on later and the one that most cleanly partitions the field. Routing is second because it is the next-hardest to retrofit. Operational preference is last because it is the most reversible — you can move a workload between SQS and NATS far more easily than you can graft replay onto a queue. The corollary is a planning rule: *interrogate replay and routing needs early, before you have built anything*, because those two are expensive to discover late.

### When the tree's answer is "it depends"

The tree gives a default, not a dogma. Several real situations push you off the obvious branch, and a good engineer knows the exceptions:

- **You need replay *and* rich routing.** These pull toward different brokers. The common resolution is to use Kafka as the system of record (replay) and project filtered subsets into RabbitMQ or onto specific topics for the routing-heavy consumers — a multi-broker architecture, covered in section nine.
- **You need replay but you are tiny and AWS-native.** Amazon Kinesis or MSK Serverless gives you log semantics without operating Kafka yourself. The tree's "Kafka" answer is about the *model*; the deployment can be managed.
- **You need rich routing but want zero ops.** SNS message-filtering policies give you a slice of attribute-based routing in a managed service — far less than AMQP, but enough for simple content routing without a cluster.
- **Latency is the dominant constraint.** If you need consistent single-digit-millisecond delivery, NATS and RabbitMQ lead; Kafka's batching and SQS's managed-service overhead both work against you.

## 8. The positioning map: throughput vs routing

The decision tree is sequential, but it helps to also see the brokers laid out in a *space*, because that reveals the structural tension at the heart of broker design. The figure below places the brokers on two axes: throughput on the vertical, routing complexity on the horizontal. The way this map works is that the upper-left and lower-right corners are crowded while the upper-right is nearly empty — and that emptiness is the whole point.

![A positioning grid mapping brokers by throughput on one axis and routing complexity on the other, showing high-throughput log brokers and rich-routing queue brokers occupying opposite regions](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-4.webp)

Read the map and the pattern jumps out. High-throughput brokers (Kafka, Pulsar, NATS core) cluster toward simple routing: they route by key or subject, nothing richer, because rich routing would cap their throughput. Rich-routing brokers (RabbitMQ) sit at moderate throughput: the per-message bookkeeping that powers AMQP routing is exactly what limits how fast a queue can go. The upper-right corner — extreme throughput *and* extreme routing richness — is nearly vacant, and that is not an accident or a gap in the market waiting to be filled. It is a consequence of the underlying mechanics. Routing every message through content-based logic means making a per-message decision; making a per-message decision means you cannot just append bytes to a file and `sendfile` them out. The two strategies are in physical tension.

This is why "just pick the fastest broker" and "just pick the most flexible broker" are both naive advice. Speed and flexibility live in opposite regions of the map. The question is never "which is best" in the abstract; it is "which corner of this map does my workload live in." A clickstream pipeline lives in the high-throughput, simple-routing corner — Kafka territory. A complex order-orchestration system with content-based routing lives in the rich-routing, moderate-throughput corner — RabbitMQ territory. Knowing *where on the map* your workload sits is more than half the decision.

## 9. The broker landscape over time

It helps to understand how we got five such different brokers, because the history explains the design choices. The figure below sketches the evolution: the messaging field did not arrive at these brokers simultaneously, and each generation was a reaction to the limits of the previous one.

![A timeline showing the messaging landscape evolving from AMQP and RabbitMQ, through Kafka's partitioned log, to SQS scaling and SNS fan-out, then Pulsar, then NATS JetStream, then managed everything](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-8.webp)

The first era was **broker-centric routing**. AMQP and RabbitMQ (mid-2000s) embodied the idea that intelligence belongs in the broker: the broker should route, filter, and manage per-message lifecycle, while clients stay dumb. This was the right model for enterprise integration and task distribution, and it is still the right model for those workloads. But it does not scale to firehose throughput, because a smart broker is a slower broker.

The second era was the **partitioned log**. Kafka (2011, born at LinkedIn) inverted the model: make the broker dumb and the clients smart. The broker just appends to a log and serves bytes; clients track their own offsets and do their own routing and filtering. This trade — give up broker-side routing, gain enormous throughput and replay — was exactly right for the explosion of event data and analytics, and it made Kafka the backbone of modern data infrastructure. The log model is dissected in [Queue vs Pub/Sub vs Log](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models).

The third era was **managed and specialized**. SQS and SNS (which predate Kafka but scaled and matured through the 2010s) offered the radical proposition that you should not run a broker at all. Pulsar (2016, from Yahoo) tried to *unify* the log and queue models while separating compute from storage, aiming to be the platform broker for multi-tenant, multi-region deployments. NATS and JetStream (JetStream landing around 2019) went the opposite direction from Pulsar's complexity, optimizing for radical operational simplicity at the edge. The most recent movement is *managed everywhere*: every broker now has a serverless or managed offering, because operational burden turned out to be the axis that most often decides adoption. The arc of the whole field bends toward "give me the model I need without the operational tax."

## 10. Worked examples

Frameworks are only as good as the decisions they produce, so let us run two concrete ones end to end. The first picks brokers for three real workloads; the second puts hard numbers on the cost-versus-ops tradeoff between self-hosted Kafka and SQS.

#### Worked example: picking a broker for three concrete workloads

You are the architect for a mid-size e-commerce company and three teams come to you in the same week with three messaging needs. Apply the decision tree to each.

**Workload A: clickstream analytics.** The web and mobile clients emit a firehose of user-interaction events — page views, clicks, scrolls, add-to-cart — roughly 200,000 events per second at peak. The data science team wants to run real-time aggregations *and* reprocess the last 30 days whenever they ship a new model. Run the tree. First question: replay? Emphatically yes — they need to reprocess 30 days of history for new models. That puts us in log territory immediately. Do they need independent compute/storage scaling, multi-tenancy, or geo-replication? Not really — it is one team, one region, one workload. So the answer is **Kafka**: high throughput handles the 200k/s firehose with room to spare, retention covers the 30-day replay window, and the stream-processing ecosystem (Kafka Streams, or Flink reading from Kafka) covers the real-time aggregation. Routing is irrelevant here — every event goes to the analytics consumers — so Kafka's weak routing costs nothing. This workload sits squarely in Kafka's sweet spot.

**Workload B: order processing with per-customer ordering.** The orders team processes purchases through a pipeline: validate, reserve inventory, charge payment, confirm. Volume is modest — maybe 500 orders per second at peak. The hard requirement is that all events for a *single customer* must be processed in order (you cannot charge before you reserve, and a cancel must not overtake the original order), but events for *different* customers are independent. They also want priority handling for express orders and a clean dead-letter path for orders that fail validation. Run the tree. Replay? Not really — they do not need to reprocess history; once an order is done it is done. So we are out of log territory. Complex routing and per-message control? Yes — priority queues for express, dead-letter exchanges for failures, and per-customer ordering. That is **RabbitMQ**, with one important design detail: per-customer ordering is achieved by using a *consistent hash exchange* (or partitioning queues by customer ID) so all of one customer's events land on the same queue and are processed in order, while different customers spread across queues for parallelism. The volume (500/s) is trivial for RabbitMQ, and its rich routing plus priority and dead-letter support is exactly the per-message control this pipeline needs. *Note the subtlety:* if they had instead needed to replay order history for audit, the answer would tilt toward Kafka with key-based partitioning by customer ID — same per-key ordering, plus replay. The replay requirement is the swing vote.

**Workload C: a fan-out notification service.** When something noteworthy happens — order shipped, price dropped, back in stock — several independent subsystems must react: send an email, send a push notification, update a recommendation model, log to the audit trail. One event, four (and growing) independent consumers, each of which must reliably receive every event even if it was briefly down. Volume is low — a few thousand events per minute. The team is small, AWS-native, and explicitly does not want to operate a broker. Run the tree. Replay? Not a hard requirement — consumers process notifications as they arrive. Complex routing? Mild — different consumers want different event types, but that is coarse attribute filtering, not AMQP-grade routing. Managed and zero-ops? Strongly yes — small team, no appetite for operations. That points to **SNS + SQS fan-out**: publish each event to an SNS topic, fan out to one SQS queue per consumer subsystem, with SNS filter policies handling the coarse "email cares about shipped, recommendations care about all" routing. Each consumer gets durable buffering in its own SQS queue (so a brief outage just means the queue backs up and drains later), at-least-once delivery, and a dead-letter queue for poison messages — all with zero operational burden. This is exactly the workload SQS/SNS was built for.

Three workloads, three different brokers, and not because of fashion — because each lives in a different region of the tradeoff space. Clickstream needs throughput and replay (Kafka). Orders need routing and per-key ordering without replay (RabbitMQ). Notifications need managed fan-out with durable per-consumer buffering (SNS/SQS). A team that forced all three onto their one favorite broker would fight the tool on at least two of them.

#### Worked example: self-hosted Kafka vs SQS at 10,000 messages per second

The most common real argument is cost: "Kafka is free, SQS charges per message, so self-host Kafka." Let us actually do the arithmetic at a sustained 10,000 messages per second, because the intuition is usually wrong in both directions.

Start with **SQS standard**. The pricing is per request, and you batch up to 10 messages per request to amortize. At 10,000 messages per second with 10-message batches, that is 1,000 send requests per second, plus receive requests, plus delete requests. Round numbers: roughly 1,000 send + 1,000 receive + 1,000 delete = about 3,000 requests per second. Over a month (about 2.6 million seconds), that is roughly 7.8 billion requests. At a representative price of \$0.40 per million requests (standard-queue pricing, with the perpetual free tier rounding off), that is about 7,800 × \$0.40 ≈ **\$3,100 per month** in request charges. Add modest data-transfer costs and call it roughly \$3,000 to \$4,000 per month. Crucially, the *operational* cost is near zero: no cluster, no on-call, no capacity planning. AWS runs it.

Now **self-hosted Kafka**. To run Kafka durably you want at least three brokers for replication and fault tolerance (plus the controller quorum, which KRaft folds into the brokers). At 10,000 messages per second of modest-sized messages, the throughput itself is trivial for Kafka — a single broker could do it — but you provision three for durability and availability. Three instances sized reasonably (say, the rough equivalent of a mid-size cloud VM each, with fast disks for the log) land somewhere around \$300 to \$600 per instance per month all-in, so call it **\$1,000 to \$1,800 per month** in pure infrastructure. On paper, that is *cheaper* than SQS at this volume. But the infrastructure bill is not the real cost.

The real cost of self-hosted Kafka is **operational**. Someone has to size partitions, configure replication and retention, monitor lag and under-replicated partitions, handle broker failures and replacements, perform rolling upgrades without downtime, plan capacity, and carry the pager. Conservatively, running a production Kafka cluster competently is a meaningful slice of an engineer's time — call it 20% to 40% of one senior engineer, which at a loaded cost of, say, \$200,000 per year is **\$40,000 to \$80,000 per year**, or roughly **\$3,300 to \$6,700 per month** in human cost. Suddenly the "cheaper" option is the *more expensive* one once you count the humans, and that is before any 2 a.m. incident.

The crossover logic falls out cleanly. At **10,000 messages per second**, SQS at roughly \$3,000 to \$4,000 per month all-in (and zero ops) is *competitive with or cheaper than* self-hosted Kafka once operations are counted — and dramatically simpler. The pure-infrastructure comparison that says "Kafka is cheaper" ignores the largest cost line. Where does self-hosting start to win? Push volume up by an order of magnitude. At **100,000+ messages per second sustained**, SQS request charges scale linearly to \$30,000+ per month, while Kafka's three-to-six broker cluster handles that volume on nearly the same hardware — the infrastructure cost barely moves, and the *fixed* operational cost is now amortized over 10× the traffic. That is the real crossover: SQS cost scales with message volume; self-hosted Kafka cost is dominated by a fixed operational floor. Below the crossover (roughly tens of thousands of messages per second, give or take, depending on your loaded engineering cost), managed wins on total cost. Far above it, self-hosted wins, *if* you have the operational maturity to run it well.

The lesson is not "SQS is cheap" or "Kafka is cheap." It is that the cost question is dominated by two terms — per-message fees that scale with volume, and a fixed operational floor that does not — and the answer flips depending on which term is larger at *your* volume. Always compute both terms. The number of teams who self-hosted Kafka to save money and instead spent a senior engineer's year on it is not small.

## 11. Anti-patterns: using the wrong broker for the job

Every one of these brokers is excellent at its sweet spot and painful outside it. The expensive mistakes in broker selection are not "I picked a bad broker" — all five are good brokers — they are "I picked a good broker for the wrong job." Here are the four anti-patterns I have seen sink projects, each a case of forcing a broker to do the thing it explicitly trades away.

The figure below contrasts the worst of these — Kafka used as a request/reply RPC bus — against Kafka used as the streaming log it is designed to be. The same broker that is a liability on the left is a powerhouse on the right; the only thing that changed is the job.

![A before and after comparison showing Kafka used as an RPC bus with high latency and reply-topic complexity versus Kafka used as a streaming log with batched appends and replay](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-3.webp)

**Anti-pattern 1: Kafka for synchronous request/reply RPC.** This is the most common and most painful. Someone wants service A to call service B and get a response, and because the company "uses Kafka for everything," they build it on Kafka: A publishes a request to a topic, B consumes it, B publishes a response to a reply topic, A consumes the response and matches it by correlation ID. Every step adds latency — the poll loop alone can add tens of milliseconds — and you have built a brittle simulation of RPC out of a log that was never meant for it. Head-of-line blocking within a partition means one slow request stalls every request behind it. The right tools for request/reply are an actual RPC framework (gRPC), or if you must use messaging, NATS request/reply (built-in, low-latency) or RabbitMQ's RPC pattern (direct-reply-to). Kafka is a log; do not make it pretend to be a phone call.

**Anti-pattern 2: RabbitMQ for event-sourcing history.** A team builds an event-sourced system on RabbitMQ, treating the queue as the event store. It works fine until they need to rebuild a projection, onboard a new read model, or audit what happened six months ago — and they discover that every event was deleted the instant it was acknowledged. There is no history to replay because a queue is consume-and-delete by design. Event sourcing *requires* a retained, replayable log; that is its defining substrate. The right tool is Kafka or Pulsar (or a purpose-built event store). RabbitMQ Streams can serve in a pinch, but a team doing serious event sourcing on classic RabbitMQ queues has built a system that cannot do the one thing event sourcing exists to do. This is the dual of anti-pattern 1: using a queue where the job demands a log.

**Anti-pattern 3: SQS standard queues where strict ordering matters.** A payments or ledger team picks SQS standard because it is cheap and infinitely scalable, and quietly assumes messages arrive in order. They do not — SQS standard provides no ordering guarantee, and out-of-order delivery is not rare; it is routine. The bug surfaces as a cancel processed before its order, or a debit applied before the credit that funds it, and it is maddening to diagnose because it is intermittent and load-dependent. The fix is SQS FIFO queues, which restore strict ordering and exactly-once processing — but FIFO caps throughput and costs more, so if you need strict ordering at high volume, SQS may be the wrong broker entirely and you want Kafka with per-key partitioning. The meta-lesson: *ordering is a requirement you must state explicitly up front*, because the default in many managed queues is no ordering, and the failure is silent.

**Anti-pattern 4: self-hosting when a managed option fits.** A small team with no platform engineers self-hosts Kafka (or Pulsar) "to save money" or "to avoid lock-in," and spends the next year fighting rebalances, under-replicated partitions, and disk-full incidents instead of building product. As the cost worked example showed, at modest volume the managed option is frequently *cheaper* once you count engineering time, and always simpler. Self-hosting is the right call when you have genuine scale (above the cost crossover), genuine operational maturity (a platform team that knows the broker cold), or a hard requirement managed services cannot meet (specific compliance, multi-cloud, extreme tuning). Absent one of those, reaching for the self-hosted cluster is paying a heavy operational tax to solve a problem a managed service already solved. The taxonomy figure below is a useful sanity check: if your need maps cleanly onto a managed offering, start there and self-host only when you can articulate why managed fails you.

![A taxonomy tree grouping brokers by their core model into queue, log, hybrid, and managed branches, placing RabbitMQ and NATS under queue and Kafka under log](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-9.webp)

That taxonomy is worth a beat of reflection, because it encodes the deepest version of the framework. Sort brokers by model first — queue, log, hybrid, managed — and the per-broker tradeoffs fall out before you run a single benchmark. A queue-model broker (RabbitMQ, NATS core) routes well and deletes on read; a log-model broker (Kafka) retains and replays but routes weakly; a hybrid (Pulsar, JetStream) tries to offer both at the cost of more parts; a managed service (SQS/SNS) trades control and per-message cost for zero operations. If you find yourself reaching for a broker whose *model* does not match your *job* — a queue where you need a log, a log where you need RPC — stop. That mismatch is the root of every anti-pattern above.

## 12. Mixing brokers in one architecture

The most sophisticated answer to "which broker?" is often "more than one." Mature architectures routinely run two or even three brokers, each handling the workload it is best at, with a clear, deliberate boundary between them. This is not indecision or sprawl — it is recognizing that a real system has multiple messaging needs that live in different regions of the tradeoff space, and trying to serve all of them with one broker means serving most of them badly.

The canonical multi-broker pattern is **Kafka as the spine, queues at the edges**. Kafka is the system of record: every significant event is published to Kafka first, giving you a single replayable log of everything that happened, which feeds analytics, stream processing, change data capture, and any future consumer that has not been invented yet. But specific consumers that need rich routing, per-message control, or zero-ops simplicity are fed from Kafka *into* the broker that suits them. A connector projects a filtered subset of Kafka events into RabbitMQ for a routing-heavy task pipeline, or into SQS for a managed fan-out to AWS Lambda functions. Kafka provides durability and replay for the whole system; the edge brokers provide the routing and ergonomics for specific consumers. You get both, with the boundary drawn exactly where the requirements change.

The matrix below captures a related dimension of mixing — the choice between self-hosted and managed deployment, per broker. The interesting observation is that every broker except SQS offers *both* a self-hosted path and a managed path, which means "self-hosted versus managed" is a deployment decision you make per broker, not a property of the broker itself. You can run managed Kafka (MSK, Confluent) and self-hosted NATS in the same architecture, choosing the deployment model that fits each workload's operational reality.

![A matrix showing self-hosted versus managed deployment options for each of the five brokers, with every broker offering a managed path and only SQS lacking a self-hosted one](/imgs/blogs/choosing-a-message-broker-kafka-rabbitmq-pulsar-nats-sqs-5.webp)

### The boundary is the design

When you mix brokers, the architecture *is* the boundary between them — where one broker ends and the next begins. A clean boundary follows the requirements: Kafka holds the replayable record because that is its strength; RabbitMQ handles the routing-heavy task pipeline because that is its strength; the bridge between them (a connector, an outbox, a consumer that re-publishes) is a small, well-understood piece of plumbing. A *messy* boundary is one drawn for organizational reasons rather than technical ones — team A likes Kafka, team B likes RabbitMQ, and now the same event flows through both for no reason anyone can articulate, with two sources of truth and a synchronization problem. The discipline is to draw the boundary where a real requirement changes (replay-here, route-there) and nowhere else.

There is a cost to mixing, and you must pay it knowingly. Two brokers means two operational models, two sets of client libraries, two monitoring stacks, two failure modes, and the integration glue between them that can itself fail or lose messages. The bridge between brokers is exactly where duplicate-delivery and ordering bugs love to hide, because you are crossing a delivery-semantics boundary. So mix when the workloads genuinely live in different regions of the tradeoff space and one broker would serve some of them poorly — but resist mixing for its own sake. Every additional broker is an additional thing that can break at 3 a.m. The right number of brokers is the smallest number that serves your workloads well, which is sometimes one, often two, and rarely more than three.

When you do bridge two brokers, the bridge itself deserves the same delivery-semantics scrutiny you would apply to any consumer-producer pair, because that is exactly what it is. A connector that reads from Kafka and writes to RabbitMQ is a consumer on one side and a producer on the other, and it inherits the failure modes of both: it can crash after publishing to RabbitMQ but before committing its Kafka offset, redelivering the message on restart, or it can commit the offset before the downstream publish is confirmed, silently dropping the message on a crash. The correct pattern mirrors what any reliable consumer does — confirm the downstream write *before* advancing the upstream cursor, and make the downstream consumer idempotent so the inevitable redelivery is harmless. Teams that treat the bridge as "just a connector, it'll be fine" are the ones who later find messages duplicated or missing precisely at the broker boundary, under load, intermittently. Draw the boundary deliberately, then engineer the crossing as carefully as you engineer the brokers it connects. A well-built bridge is boring and reliable; a careless one is the quiet source of the gnarliest distributed-systems bugs you will ever debug.

## Case studies and war stories

Abstract frameworks land harder when attached to real incidents. Here are three patterns drawn from production, each teaching one of the lessons above.

### The Kafka-as-RPC latency mystery

A platform team standardized on Kafka and, over time, services started using it for request/reply: a service would publish a request to a topic and wait for a response on a reply topic. It worked in development and under light load. In production at peak, the p99 latency of these request/reply calls ballooned to seconds, and nobody could find a slow service to blame. The cause was structural, not a bug in any one service. Head-of-line blocking within partitions meant that one slow handler stalled every request queued behind it in the same partition, and the poll-loop consumer model added baseline latency to every call. The fix was not to tune Kafka — it was to *stop using Kafka for request/reply*. They moved the synchronous calls to gRPC and kept Kafka for the genuinely asynchronous event flows. The lesson: Kafka's latency under request/reply load is a property of the model, not a misconfiguration, and no amount of tuning fixes a model mismatch. Pick the broker whose model matches the interaction pattern.

### The RabbitMQ replay that did not exist

An event-sourced billing system was built on RabbitMQ. For two years it worked. Then a new finance requirement landed: rebuild a year of account balances from the event history to satisfy an audit. The team went looking for the event history and found nothing — every event had been deleted on acknowledgement the moment it was consumed, because that is what a queue does. There was no log to replay, no offset to rewind, no retained record. They had been doing "event sourcing" on a substrate that, by design, does not retain events. The recovery was expensive: reconstruct what history they could from database snapshots and downstream side effects, then re-platform onto Kafka so future history would be retained and replayable. The lesson: event sourcing requires a log, full stop. A queue can carry events, but it cannot *be the event store*, because its defining behavior is to delete what it delivers. State the replay requirement before you choose the broker, not two years after.

### The SQS ordering bug that came and went

A team built an inventory-adjustment service on SQS standard queues — cheap, scalable, zero-ops, perfect. In testing and at low load, everything was ordered correctly. In production at peak, inventory counts occasionally went wrong in ways that healed themselves on retry, which made the bug nearly impossible to reproduce. The root cause was that SQS standard provides no ordering guarantee, and at high load, a decrement occasionally arrived before the increment that should have preceded it, briefly driving a count negative before a later message corrected it. The intermittency was the tell: ordering violations are load-dependent and rare, so they slip through testing and surface only in production. The fix was to move the ordering-sensitive messages to an SQS FIFO queue keyed by product ID, accepting FIFO's throughput cap for those messages while keeping standard queues for the order-insensitive ones. The lesson: never *assume* ordering. If a workload depends on order, demand a broker that guarantees it, and confirm the guarantee's scope (per-key, per-partition, global) matches what your logic assumes.

## When to reach for each broker (and when not to)

Here is the decisive recommendation, broker by broker, distilled to the cases where each is clearly right and clearly wrong.

**Reach for Kafka when** you need high-throughput event streaming, a replayable log as a system of record, stream processing, change data capture, or log aggregation, *and* you have (or can build) the operational maturity to run a stateful cluster. **Do not reach for Kafka when** you need synchronous request/reply, content-based per-message routing, or a tiny zero-ops deployment — it is heavy, batch-biased, and routes only by key.

**Reach for RabbitMQ when** you need low-latency task distribution, complex content-based routing, per-message control (priorities, TTL, dead-lettering), or RPC-style work queues, at moderate throughput. **Do not reach for RabbitMQ when** you need replayable history (it deletes on ack) or firehose throughput (its per-message bookkeeping caps it) — those are exactly what it trades away for routing power.

**Reach for Pulsar when** you specifically need independent scaling of compute and storage, native multi-tenancy, geo-replication, or unified queue-and-log semantics over the same data — and you have the operational appetite for its larger component count. **Do not reach for Pulsar when** Kafka's simpler architecture and deeper ecosystem already meet your needs; you would be paying for complexity you do not use.

**Reach for NATS/JetStream when** you want fast, simple messaging for microservices or the edge, native request/reply, and the lowest possible operational burden among self-hosted brokers, with optional durability via JetStream. **Do not reach for NATS when** you need a deep stream-processing ecosystem, an enormous connector library, or the maximal production track record of the most-deployed log broker.

**Reach for SQS/SNS when** you are AWS-native, want zero operational burden, need effectively infinite elastic scale for task queues or managed fan-out, and your ordering needs are either none or fit within FIFO's limits. **Do not reach for SQS when** you need replay (it has none, 14-day max retention), strict high-throughput ordering (FIFO caps throughput), multi-cloud portability, or low cost at very high sustained volume (per-request fees add up).

The unifying principle: every broker is excellent inside its sweet spot and a liability outside it. The skill is not knowing which broker is "best" — none is — but knowing which region of the tradeoff space your workload lives in and matching the broker whose strengths cover that region and whose weaknesses you can tolerate.

## Key takeaways

- **Score the right axes.** Throughput is the most benchmarked and least often decisive axis. The axes that usually decide it are replay, routing complexity, ordering, and operational burden. Identify your dominant constraint before comparing brokers.
- **No broker wins every axis, by physics.** Rich routing and high throughput are in genuine tension — content-based per-message routing precludes the dumb-append, zero-copy path that makes log brokers fast. Choosing a broker is choosing which sacrifices you can live with.
- **Replay is the sharpest dividing line.** Ask it first. If you need to reprocess history, you need a log (Kafka or Pulsar); no queue-shaped broker (RabbitMQ, NATS core, SQS) retains messages after acknowledgement. This requirement is the most expensive to retrofit.
- **Match the model to the interaction.** Kafka for streaming, RabbitMQ for routing and task work, NATS for fast microservice messaging and RPC, SQS/SNS for managed fan-out, Pulsar for multi-tenant platform scale. Forcing a model mismatch (Kafka for RPC, RabbitMQ for event sourcing, SQS for strict ordering) is the root of every anti-pattern.
- **Count operational cost, not just infrastructure cost.** Self-hosted "free" brokers carry a fixed operational floor that often exceeds a managed service's per-message fees at modest volume. Managed wins below the cost crossover; self-hosted wins far above it. Compute both terms at your volume.
- **State ordering requirements explicitly.** Many managed queues default to no ordering, and ordering violations are intermittent and load-dependent, so they slip past testing. If your logic assumes order, demand a guarantee and confirm its scope (per-key vs global).
- **Mixing brokers is mature, not messy — if the boundary is principled.** Use Kafka as the replayable spine and feed routing-heavy or zero-ops consumers from it into RabbitMQ or SQS. Draw the boundary where a real requirement changes, and pay the integration cost knowingly. The right number of brokers is the smallest that serves your workloads well.
- **Default to managed, self-host with a reason.** Reach for the managed option first; self-host only when you have genuine scale, genuine operational maturity, or a hard requirement managed cannot meet. The teams that self-hosted to save money and spent an engineer-year are not rare.

## Further reading

- [Queue vs Pub/Sub vs Log: three messaging models](/blog/software-development/message-queue/queue-vs-pubsub-vs-log-three-messaging-models) — the conceptual map this broker comparison lives inside.
- [Kafka deep dive: log segments, page cache, and storage](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage) — why Kafka's append-only log is fast and why it routes only by key.
- [Kafka replication, ISR, acks, and durability](/blog/software-development/message-queue/kafka-replication-isr-acks-durability) — the durability mechanics behind the replay guarantee.
- [RabbitMQ AMQP exchanges, bindings, and routing](/blog/software-development/message-queue/rabbitmq-amqp-exchanges-bindings-routing) — the routing engine that defines RabbitMQ's sweet spot.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the log data model in depth.
- [RabbitMQ production architecture and scaling](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) — operating RabbitMQ at scale.
- Apache Kafka documentation (kafka.apache.org), Apache Pulsar documentation (pulsar.apache.org), NATS documentation (docs.nats.io), and the AWS SQS and SNS developer guides — the authoritative references for each broker's exact guarantees and configuration.
