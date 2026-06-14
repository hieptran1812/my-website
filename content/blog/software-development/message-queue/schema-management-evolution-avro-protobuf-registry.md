---
title: "Schema Management and Evolution: Avro, Protobuf, and the Schema Registry"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "When producers and consumers are decoupled, the message schema is the API between them. Learn how to serialize compactly with Avro and Protobuf, how a schema registry ships IDs instead of inline schemas, what backward, forward, and full compatibility actually let you do, and the precise rules for evolving a schema without paging anyone at 3am."
tags:
  [
    "message-queue",
    "schema-registry",
    "avro",
    "protobuf",
    "serialization",
    "schema-evolution",
    "kafka",
    "distributed-systems",
    "event-driven",
    "data-contracts",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/schema-management-evolution-avro-protobuf-registry-1.webp"
---

There is a particular kind of outage that has no stack trace at the site of the crime. A data engineer on the orders team adds a field to the order event — `loyalty_tier`, a harmless string, ships on a Tuesday afternoon. Nothing breaks in their service. Their tests pass. The event flows. Three days later, the fraud team's consumer starts throwing deserialization errors on every message, the analytics warehouse silently drops a column, and a downstream team in another timezone gets paged because their nightly job is now reading garbage. Nobody changed the fraud consumer. Nobody changed the warehouse. The only thing that changed was a schema three teams away — and that schema, it turns out, was the actual API between all of them. They just never wrote it down as one.

This is the central, uncomfortable truth of decoupled systems. When you put a message broker between a producer and a consumer, you decouple them in time, in space, and in deployment schedule — which is exactly the point, and exactly the win. But you do not decouple them in *meaning*. The producer still has to write bytes that the consumer can read, and the agreement about what those bytes mean — which fields exist, what types they are, what is optional — is a contract that spans services and survives every deploy. The message schema **is** the API. The difference is that a REST API has a version in the URL and a team that owns it, while a message schema is invisible, implicit, and owned by nobody until the day it breaks. This post is about making that contract explicit, compact, and safe to evolve.

The figure below is the whole problem and the whole solution in one frame. On the left: no governance, where a producer renames a field and a consumer three hops away fails to deserialize at 3am with no warning anyone could have acted on. On the right: a schema registry, which checks every proposed schema against a compatibility rule *before* it ships, rejects the breaking change at publish time, and turns a production incident into a failed CI step. Everything in this post is a zoom into one side of that picture.

![A before and after comparison showing an ungoverned system where a producer field rename breaks a downstream consumer at runtime versus a schema registry that validates evolution and rejects breaking changes before they ship](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-1.webp)

By the end you will be able to do four concrete things. You will pick a serialization format — JSON, Avro, Protobuf, or Thrift — with reasons, not vibes. You will explain exactly how a schema registry lets you ship a tiny schema ID per message instead of the full schema, and do the bandwidth math that makes this matter at scale. You will name the compatibility modes — backward, forward, full, and their transitive variants — and say which one lets you safely upgrade producers first versus consumers first. And you will know the precise menu of safe and unsafe schema changes, so that the next time you add a field you do it with a default and a clear conscience, and the next time someone proposes removing a required field you can point at the rule and the registry that will stop them. This is the schema and serialization installment of the series; it builds directly on the [anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers), where we established that serialization is a producer responsibility and the format is a contract with every consumer present and future.

## 1. The schema is the contract between producer and consumer

Let us be precise about what a schema is, because the word gets used loosely. A **schema** is a machine-readable description of the structure of a message: the set of fields, the type of each field, which fields are required versus optional, and — critically — what a reader should do when a field it expected is missing or a field it did not expect is present. That last part is what separates a schema from a mere data shape. A data shape tells you what a *correct* message looks like today. A schema, done well, also tells you how to read messages written by code that is older or newer than yours. That forward-and-backward reading behavior is the entire game in a decoupled system, because producers and consumers are never on the same version at the same time.

Why are they never on the same version? Because that is the deal you signed when you introduced the broker. In a synchronous RPC call, the caller and callee are coupled in time: the call fails immediately if the contract is violated, and you find out in the deploy that caused it. In an asynchronous message system, the producer writes a message now and a consumer reads it minutes, hours, or — in a replayable log like Kafka — *months* later. During that window, both sides deploy independently and repeatedly. A message written by producer v5 might be read by consumer v3 (which has not caught up yet) and also by consumer v9 (which was deployed to handle a future format). The schema has to make all of those combinations work, or at least make the dangerous ones fail loudly at the registry instead of silently at runtime.

### The four ways a contract can be violated

There are exactly four interesting things that can go wrong when a producer's schema and a consumer's schema disagree, and naming them sharpens everything that follows.

The first is a **missing field**: the consumer expects a field that the message does not contain. This happens when a consumer running new code reads a message written by an old producer that did not have the field yet — or when a producer removes a field the consumer still depends on. The second is an **extra field**: the message contains a field the consumer's schema does not know about. This happens when a new producer adds a field and an old consumer reads it. A well-designed format makes the extra field harmless (ignore it); a badly-designed one chokes. The third is a **type mismatch**: a field that used to be an integer is now a string, or a single value became a list. This is almost always fatal, because the bytes for an integer and the bytes for a string are not interchangeable, and there is no safe default. The fourth is a **renamed field**, which is the cruelest because it looks innocent. A rename is, to the serializer, a remove plus an add: the old name vanishes (missing field for anyone keyed on it) and a new name appears (extra field for everyone else). Renaming `customerId` to `customer_id` can take down a pipeline as thoroughly as deleting the field.

Every schema-evolution rule we cover later is just a policy about which of these four situations is allowed and how each format handles it. Hold these four in your head — missing, extra, type-changed, renamed — and the rest of the post is bookkeeping.

### Where the contract physically lives

In the most common, naive setup, the contract lives nowhere. The producer serializes with whatever fields it happens to have today, the consumer parses with whatever fields it happens to expect today, and the agreement between them exists only as an oral tradition passed between two teams in a Slack channel. This works right up until it does not, and when it fails it fails in production, asymmetrically: the team that made the change is fine, and the team that broke is the one that gets paged. That asymmetry — the breaker is safe, the broken is paged — is the single most corrosive dynamic in event-driven organizations, because it removes the natural feedback loop. The person who would learn from the mistake never sees it.

The fix is to give the contract a home: a place where the schema is written down explicitly, versioned, and checked. That home can be a `.proto` file in a shared repository, an Avro `.avsc` file published to an artifact store, or — the subject of the back half of this post — a running **schema registry** that producers and consumers both talk to. The mechanism varies; the principle does not. Make the contract a first-class, owned, versioned artifact, and make a machine check changes to it, because humans reviewing a one-line field rename will wave it through every single time.

## 2. Serialization formats: JSON, Avro, Protobuf, Thrift

Serialization is the act of turning your in-memory object into a byte array, because — as we established in the anatomy post — a broker stores bytes and nothing else. The format you choose is not a plumbing detail; it determines your payload size, your CPU cost, whether you can evolve safely, and how much tooling you inherit. There are four formats worth knowing, and they cluster into two philosophies: schemaless text (JSON) and schema-driven binary (Avro, Protobuf, Thrift). The matrix below compares them on the four axes that actually decide the choice, and the sections after it explain each cell.

![A matrix comparing JSON, Avro, Protobuf, and Thrift across payload size, schema model, evolution safety, and tooling maturity showing binary schema formats winning on size and evolution](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-3.webp)

### JSON: schemaless, verbose, forgiving, fragile

JSON is the default for a reason: it has zero tooling cost. Every language can parse it, you can read it with your eyes, and you can `curl` an endpoint and pipe it through `jq`. For a small system or a debugging session, this is genuinely valuable, and I will not pretend otherwise. But JSON has two structural problems at scale. The first is **size**. JSON ships the field names with every single message — the string `"customer_id"` appears in the bytes of every record, alongside the braces, quotes, and commas. A record that is really just four numbers and two short strings can easily be 800 bytes of JSON, most of which is repeated field names and punctuation.

The second problem is more dangerous: JSON is **schemaless**, which sounds like freedom and is actually a liability. There is no agreed-upon definition of what fields a message must have or what types they are. The "schema" of a JSON message is whatever the producer happened to emit, and the consumer discovers it by reading the bytes and hoping. JSON is *forgiving* in the moment — a missing field just parses as `null`, an extra field is usually ignored, and nothing crashes immediately — which is exactly why it is *fragile* over time. The forgiveness hides the contract violation until some code three hops downstream does `order.total.toFixed(2)` on a `null` and throws. JSON does not break at the boundary where the mistake was made; it breaks deep in someone else's code, far from the cause. You can bolt a schema onto JSON with JSON Schema, and it helps, but you are now maintaining a schema separately from a format that does not enforce it, which is the worst of both worlds.

### Avro: the schema travels with the data, logically

Avro takes the opposite stance: a message is meaningless without its schema, and the schema is a first-class, explicit document (an `.avsc` JSON file describing the record). The clever part is how Avro uses two schemas at read time. When you serialize, Avro uses the **writer's schema** — the schema the producer had. When you deserialize, Avro uses both the writer's schema *and* the **reader's schema** — the schema the consumer expects — and performs *schema resolution* to reconcile them. If the writer had a field the reader does not want, Avro drops it. If the reader wants a field the writer did not have, Avro fills in the reader's declared **default**. This two-schema resolution is the mechanism that makes Avro the format *designed* for evolution: the reader explicitly tells Avro how to interpret older or newer data.

The catch is that schema resolution requires the reader to know the writer's schema, not just its own. The schema does not literally travel inside each message — that would defeat the size win — but it must travel *logically*, which in practice means a schema registry the reader can consult. Avro is compact because, knowing the schema, it writes only the values in field order with no field names at all — that 800-byte JSON record becomes roughly 70 bytes of Avro. Avro's evolution rules are precise and built into the format: fields can be added if they have defaults, removed if they had defaults, and the reader-default mechanism handles the rest. It is the strongest evolution story of the four, which is why Confluent's registry and the Kafka ecosystem grew up around it.

### Protobuf: field numbers are the contract

Protocol Buffers (Protobuf) makes a different bet: instead of resolving two schemas at read time, it pins the contract to **field numbers**. In a `.proto` file, every field gets a number — `string customer_id = 1;` — and that number, not the name, is what gets written on the wire. The serialized bytes are a sequence of (field number, wire type, value) tuples. This has a beautiful consequence: you can rename `customer_id` to `customer_identifier` in your `.proto` and nothing breaks, because the wire format never carried the name — it carried the number `1`. Adding a field means adding a new number; old readers see a number they do not recognize and skip it; new readers reading old data find the number absent and use the field's zero-value default.

Protobuf's evolution model is therefore "append-only on field numbers": never reuse a retired number, never change a field's type, and you can add and (carefully) remove fields freely. It is slightly larger on the wire than Avro for some shapes because it tags each field with its number and wire type inline, but it is still compact — that same record is roughly 75 bytes. Protobuf's killer advantage is **tooling**: it is the serialization format of gRPC, it has first-class code generation for a dozen languages, and the `.proto` file is a clean, readable contract you check into a repo. If your organization already speaks gRPC, Protobuf in your message bus is a natural and well-supported choice, and the schema registry supports it alongside Avro.

### Thrift: the elder statesman

Apache Thrift, born at Facebook, predates Protobuf in spirit and is conceptually similar: an interface definition language (IDL) with field IDs, compact binary encoding, and code generation across many languages. It bundles serialization *and* an RPC framework together, which was its original selling point. Technically it is sound and its binary format is competitive on size. In practice, the ecosystem has consolidated around Protobuf (carried by gRPC and Google's gravity) and Avro (carried by the Kafka and Hadoop data ecosystem), so Thrift today is mostly seen in older systems and a few large shops that standardized on it years ago. If you are starting fresh, you will almost certainly choose Avro or Protobuf; I mention Thrift so you recognize it and understand it is the same family — IDL plus field IDs plus binary — not a different paradigm.

Here is the honest one-paragraph decision rule. Use **JSON** for low-volume, human-facing, or debugging-heavy paths where tooling-free convenience beats every other concern. Use **Avro** when you are deep in the Kafka and Confluent ecosystem, want the strongest evolution semantics, and are comfortable with a registry being mandatory. Use **Protobuf** when you already live in gRPC, want field-number stability and broad cross-language codegen, and value rename-safety out of the box. Reach for **Thrift** essentially only if you have already standardized on it. The rest of this post uses Avro for the registry mechanics because that is where the registry model is most explicit, and calls out Protobuf differences where they matter.

## 3. Why binary schemas beat schemaless JSON at scale

The case for binary is usually made on "performance," which is true but underspecified. Let us make it concrete with the two costs that actually move: bytes on the wire (and on disk, and across replication) and CPU to encode and decode. The wire cost is the dramatic one, and it compounds in a way that surprises people, because a message system replicates and stores every byte you send, often three times for durability, and then ships it to every consumer.

#### Worked example: 800-byte JSON versus Avro plus a 5-byte header at 100k msg/s

Take a realistic order event: an order ID, a customer ID, a timestamp, an amount, a currency, and a status. As pretty-printed-ish JSON with the field names spelled out, this is comfortably **800 bytes** per message — I have measured production order events north of a kilobyte once you add a few nested fields and addresses. The same record in Avro, where the field names live in the schema and the bytes carry only the values in order, is about **70 bytes** of payload. The registry adds a **5-byte header** to each message: one magic byte plus a 4-byte schema ID. So the Avro-on-the-wire size is roughly **75 bytes**.

Now run it at a steady **100,000 messages per second**, a normal rate for a busy topic at a mid-sized company. The JSON stream is 800 bytes times 100,000, which is **80 megabytes per second**, or about 640 megabits per second of producer egress before you even replicate it. The Avro stream is 75 bytes times 100,000, which is **7.5 megabytes per second**. That is a **10.7x reduction** on the wire. Multiply by the durability factor: with a replication factor of 3, the broker writes and ships every byte three times across the cluster, so the JSON workload is moving 240 MB/s of internal replication traffic versus 22.5 MB/s for Avro. Over a day, the JSON topic ingests about 6.9 terabytes; the Avro topic ingests about 650 gigabytes. Same information. The difference — roughly 6.2 TB per day, every day — is pure overhead you pay in cloud egress charges, in disk you provision for retention, and in the page cache you are trying to keep hot for your consumers. The bandwidth and storage you save is not a rounding error; at this scale it is a line item.

The size win exists because the schema is **factored out**. JSON repeats the structure in every message; binary-with-schema states the structure once, in the registry, and ships only the data. The figure below shows where that schema physically lives relative to the bytes you put on the wire — the full schema sits once in the registry underneath, and each record carries only a tiny ID header over a compact payload.

![A layered stack diagram showing a registry holding the full schema once at the bottom, then a magic byte, then a four-byte schema ID, then the compact serialized payload on top](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-7.webp)

### The CPU cost cuts the same direction

You might expect binary to cost more CPU because it is "more complex," but the opposite is usually true. JSON parsing is deceptively expensive: the parser scans every byte looking for delimiters, handles string escaping, parses numbers from text (turning `"123.45"` into a float is real work), and allocates a map of string keys. Binary formats skip all of that — they read fixed-layout or length-prefixed fields directly into typed values with no text parsing and far fewer allocations. In typical benchmarks, Avro and Protobuf encode and decode several times faster than JSON for the same record, and they generate far less garbage, which matters enormously in a JVM consumer where GC pressure under high throughput is a real operational concern. So binary wins on bytes *and* on CPU *and* on memory pressure. The only axis JSON wins is "I can read it with my eyes," and for that you keep a small tool that decodes a sampled message on demand rather than paying the tax on every message forever.

### The catch you are buying

None of this is free. Binary-with-schema introduces a hard dependency: you cannot read a message without its schema. A raw Avro payload pulled off a topic is an opaque blob until you resolve its schema. This means your registry is now on the critical path for *understanding* data — not for producing or consuming the bytes (the broker happily moves bytes it cannot interpret), but for any code that needs to deserialize. That dependency is the price of the size and safety wins, and the back half of this post is about operating that dependency responsibly so it never becomes the thing that takes you down. Cross-reference the [Kafka deep dive on storage](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage) for why the payload-size reduction also directly improves your page-cache hit rate and sequential-read throughput on the broker.

### The amplification that makes the size win larger than it looks

The 10x payload reduction is the headline, but the real-world saving is bigger because every byte you write is *amplified* on its way through a durable message system, and the amplification factor multiplies the difference. Trace one byte of payload. The producer sends it to the partition leader (1x egress from the producer). The leader writes it to its local log (1x disk write). With replication factor 3, the leader ships it to two followers, each of which writes it to their own log — so that one byte becomes 2 more network crossings and 2 more disk writes. Then every consumer group that reads the topic fetches the byte once per group: a topic with five independent consumer groups serves the byte five more times. A single logical byte, by the time it has been durably stored and delivered to everyone who wants it, has been written to disk three times and crossed the network seven or eight times.

Now apply that to the JSON-versus-Avro gap. The 725-byte-per-message difference between JSON and Avro is not a 725-byte difference in the system — it is 725 bytes times the amplification factor. At replication factor 3 and five consumer groups, that is roughly 725 bytes turned into something like 5–6 kilobytes of total disk-and-network work per message saved. At 100k msg/s that is on the order of 500–600 MB/s of internal cluster work you simply do not do by choosing the compact format. This is why the size argument is not pedantry: the saving lands on your most contended resources — inter-broker replication bandwidth, disk write throughput, and the page cache you are fighting to keep hot — and it lands multiplied. A team that switches a high-fanout topic from JSON to Avro routinely sees broker CPU and network utilization drop by a third, not because the broker got faster but because it is moving a third as many bytes.

## 4. The schema registry: IDs instead of inline schemas

Here is the obvious-in-hindsight idea that makes binary serialization practical. If the consumer needs the writer's schema to decode, and shipping the full schema with every message would erase the size win, then ship a *reference* to the schema instead — a small integer ID — and let both sides resolve that ID to the full schema through a shared service. That service is the **schema registry**. Confluent's Schema Registry is the canonical implementation in the Kafka world, but Apicurio, AWS Glue Schema Registry, and others follow the same model. It is a small, stateful HTTP service that stores schemas, assigns each a globally unique integer ID, and answers two questions all day: "here is a schema, what is its ID and is it compatible?" and "here is an ID, what is the schema?"

The flow is worth tracing end to end, because once you see it the whole architecture clicks. The figure below lays it out: the producer registers its schema with the registry and gets back an ID (say 42); it then serializes the record and prepends the ID; the broker stores the resulting bytes without ever knowing or caring what they mean; the consumer reads the bytes, extracts the ID from the header, asks the registry for the schema behind ID 42 (and caches the answer so it asks once, not once per message), and uses that schema to deserialize. The full schema crossed the network exactly twice ever — once when the producer registered it, once when each consumer first fetched it — and not once per message.

![A grid diagram showing the schema registry flow where a producer registers a schema to get an ID, embeds the ID with the payload, the broker stores only bytes, and the consumer fetches the schema by ID from a cache to deserialize into a typed record](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-2.webp)

### The wire format: magic byte, schema ID, payload

Concretely, the Confluent wire format prepends a 5-byte header to the serialized payload: one **magic byte** (currently `0x00`, a format-version marker so the framing can evolve later) followed by a **4-byte big-endian schema ID**. After that comes the Avro (or Protobuf) payload. That is the entire framing. A consumer's deserializer reads the magic byte to confirm the format, reads the next four bytes as the schema ID, looks up the schema (from its local cache or, on a miss, from the registry), and hands the remaining bytes plus the schema to the Avro decoder. Five bytes of overhead per message buys you the entire compatibility-checked, evolution-safe contract. At the 100k msg/s from the worked example, those 5 bytes are 500 KB/s — utterly negligible next to the 72.5 MB/s you saved by not shipping JSON.

### Registration and the subject

When a producer registers a schema, it registers it under a **subject** — a named scope for the schema's evolution history. By the default `TopicNameStrategy`, the subject for a topic `orders` is `orders-value` (and `orders-key` for the key). The subject is what the registry tracks versions and compatibility against: version 1, version 2, version 3 of the `orders-value` subject form a lineage, and the compatibility mode you set is enforced *within that subject's history*. This matters because the subject, not the topic, is the unit of governance. You can have multiple schemas (different IDs) tied to one subject's version history, and you choose the subject naming strategy to control how strictly schemas are scoped — per topic, per record type, or per topic-and-record.

Here is a producer registering and serializing with the Confluent Avro serializer, which handles the registration and header framing for you:

```python
from confluent_kafka import SerializingProducer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroSerializer

schema_str = """
{
  "type": "record",
  "name": "Order",
  "namespace": "com.shop.events",
  "fields": [
    {"name": "order_id",    "type": "string"},
    {"name": "customer_id", "type": "string"},
    {"name": "amount_cents","type": "long"},
    {"name": "currency",    "type": "string", "default": "USD"},
    {"name": "status",      "type": "string"}
  ]
}
"""

sr_client = SchemaRegistryClient({"url": "http://schema-registry:8081"})
# The serializer registers the schema under subject "orders-value" on first use,
# caches the assigned ID, and prepends the 5-byte header to every record.
avro_serializer = AvroSerializer(sr_client, schema_str)

producer = SerializingProducer({
    "bootstrap.servers": "broker:9092",
    "key.serializer": lambda k, ctx: k.encode("utf-8"),
    "value.serializer": avro_serializer,
})

producer.produce(
    topic="orders",
    key="order-1001",
    value={
        "order_id": "order-1001",
        "customer_id": "cust-77",
        "amount_cents": 4999,
        "currency": "USD",
        "status": "PLACED",
    },
)
producer.flush()
```

And the consumer side, where the deserializer reads the ID from the header and fetches the schema transparently:

```python
from confluent_kafka import DeserializingConsumer
from confluent_kafka.schema_registry import SchemaRegistryClient
from confluent_kafka.schema_registry.avro import AvroDeserializer

sr_client = SchemaRegistryClient({"url": "http://schema-registry:8081"})
# No schema string needed here: the deserializer pulls the writer's schema
# by the ID embedded in each message, then resolves against the reader's.
avro_deserializer = AvroDeserializer(sr_client)

consumer = DeserializingConsumer({
    "bootstrap.servers": "broker:9092",
    "group.id": "fraud-scoring",
    "key.deserializer": lambda k, ctx: k.decode("utf-8") if k else None,
    "value.deserializer": avro_deserializer,
    "auto.offset.reset": "earliest",
})
consumer.subscribe(["orders"])

while True:
    msg = consumer.poll(1.0)
    if msg is None:
        continue
    if msg.error():
        print(f"consume error: {msg.error()}")
        continue
    order = msg.value()  # a dict, decoded via the schema fetched by ID
    score_for_fraud(order)
```

Notice what the consumer does *not* do: it never hardcodes a schema. It declares its expectations through the schema it was generated or configured against, and the deserializer reconciles that against whatever writer's schema each message references. That reconciliation is schema resolution, and it is what makes the same consumer binary read messages from three different producer versions without redeploying.

### The full round trip, end to end

It is worth tracing the complete serialize-and-deserialize round trip as a single pipeline, because the registry's role at each end is easy to lose in the per-message detail. The pipeline below is that round trip: the producer serializes the record and stamps it with the schema ID, the framing prepends the 5-byte header, the broker appends the resulting bytes without interpreting them, and on the read side the consumer resolves the ID through the registry before the decoder turns bytes back into a typed object. The way this works is that the schema metadata on the wire is *only* the ID — everything else needed to interpret the bytes is fetched out of band and cached — so the broker stays a pure byte mover and the registry stays off the per-message hot path.

![A pipeline showing serialize with a schema ID, prepend a five-byte header, broker append of bytes only, resolve the ID through the registry, and deserialize into a typed object](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-5.webp)

The asymmetry between the two ends is worth dwelling on. On the produce side, the registry interaction happens *once per schema*, at registration, and from then on the producer holds the ID and stamps it on every record with no further registry contact. On the consume side, the registry interaction happens *once per distinct ID a consumer encounters*, and thereafter the schema is served from the client's local cache. So a producer emitting a billion messages a day under one schema touches the registry once; a consumer reading those billion messages touches the registry once. The registry's request volume is proportional to the number of *distinct schemas in flight*, not to message throughput — which is precisely why a single modest registry instance comfortably serves a cluster moving millions of messages per second. This decoupling of registry load from message load is the property that makes the whole design scale, and it is the first thing to verify when someone worries that "a central registry will be a bottleneck": it is not on the hot path, by construction.

One more subtlety that trips people up: the broker genuinely does not know or care that the bytes are Avro. To the broker, a registry-encoded message is an opaque key and an opaque value, exactly like any other record. This means you can run the registry alongside an existing cluster without changing a single broker setting — the framing is entirely a client-side convention. It also means a misconfigured consumer that bypasses the deserializer and reads raw bytes will get the magic byte and ID header as the first five bytes of every value, which is a common "why does my message start with garbage" confusion the first time someone debugs with a plain console consumer. The fix is to use the registry-aware console consumer, which strips and resolves the header for you.

## 5. Compatibility modes: backward, forward, full, transitive

The registry's superpower is not storing schemas — a database can do that. It is *refusing* to store a schema that would break the contract. When you register a new version of a subject, the registry checks it against the configured **compatibility mode** and rejects it if it violates the rule. Getting these modes right is the difference between a registry that protects you and a registry that is a write-only graveyard. The tree below organizes the four modes and their transitive variants; each one answers a different question about which side can read whose data.

![A tree diagram of compatibility modes branching into backward, forward, and full, each splitting into a transitive variant that extends the compatibility check across all prior versions rather than only the latest](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-4.webp)

### Backward compatibility: new schema reads old data

**Backward** compatibility means a **new schema can read data written with the old schema**. The direction to fix in your head: the *reader* is new, the *data* is old. This is the most common default, and the reason is operational: it matches the most common upgrade order. If your new consumer code (with the new schema) can read all the messages already sitting on the topic — including the old ones written before the change — then you can deploy the new consumer safely while old messages are still in flight or in the retention window. The changes that preserve backward compatibility are: **adding a field with a default** (the new reader supplies the default when reading old data that lacks the field) and **removing a field** (the new reader simply ignores data that the old schema had but the new one does not need). What backward compatibility forbids is adding a *required* field with no default, because then the new reader hits old data missing that field and has nothing to fall back on.

### Forward compatibility: old schema reads new data

**Forward** compatibility is the mirror image: an **old schema can read data written with the new schema**. The *reader* is old, the *data* is new. This matters when you want to upgrade the *producer* first and let consumers catch up on their own schedule. If a producer starts emitting the new format and your not-yet-upgraded consumers can still read those new messages with their old schema, you have forward compatibility. The safe changes flip relative to backward: **adding a field** is forward-safe (the old reader ignores the field it does not know about), and **removing a field that had a default** is forward-safe (the old reader, encountering data without that field, uses its default). What forward compatibility forbids is removing a field that the old reader requires with no default — because then the old reader hits new data missing a field it cannot do without.

### Full compatibility: both directions

**Full** compatibility requires *both* backward and forward — every change must be safe in both reading directions simultaneously. The practical consequence is restrictive but liberating: the only changes that are fully compatible are **adding an optional field with a default** and **removing an optional field that had a default**. Under full compatibility you can upgrade producers and consumers in *any order*, in *any mix*, with zero coordination, because every version can read every other version's data. That is the gold standard for a contract shared across many independently-deployed teams, and it is the mode I default to for any event that more than two teams consume. The cost is that you can only ever add and remove optional-with-default fields; you give up the ability to ever make a field required through evolution.

### Transitive variants: across all versions, not just the latest

By default, the registry checks a new schema only against the **latest** registered version. The **transitive** variants — `BACKWARD_TRANSITIVE`, `FORWARD_TRANSITIVE`, `FULL_TRANSITIVE` — check the new schema against **every** previous version in the subject's history. Why does this matter? Because non-transitive compatibility is not actually transitive: v3 can be backward-compatible with v2, and v2 backward-compatible with v1, yet v3 *not* backward-compatible with v1. If you have a long retention window — say 90 days, or an event-sourcing log you replay from the beginning of time — a consumer might encounter a v1 message at any moment, and pairwise-latest checking does not protect you against it. Transitive modes guarantee the property you probably assumed you had: any version can read any other version's data, across the entire history. The cost is that they are stricter and occasionally reject a change that would have been fine for all *live* data. For replayable logs and long retention, pay the cost; transitive is the honest choice. This connects directly to [event sourcing and CQRS on a commit log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log), where the entire history is replayable and a non-transitive check is a trap waiting years to spring.

The table summarizes which changes survive which mode:

| Change | Backward | Forward | Full |
| --- | --- | --- | --- |
| Add optional field (with default) | safe | safe | safe |
| Add required field (no default) | breaks | safe | breaks |
| Remove field that had a default | safe | safe | safe |
| Remove required field (no default) | safe | breaks | breaks |
| Rename field (no alias) | breaks | breaks | breaks |
| Change a field's type | breaks | breaks | breaks |

## 6. Which mode lets you upgrade which side first

This is the question that turns compatibility theory into a deployment runbook, and it is where most engineers get tangled, so let us make it mechanical. The rule is short: **backward compatibility lets you upgrade consumers first; forward compatibility lets you upgrade producers first; full lets you upgrade either side first, in any order.** Now the reasoning, because you should be able to derive it rather than memorize it.

### Why backward means consumers first

Backward compatibility guarantees the new schema reads old data. So if you deploy the new consumer (carrying the new schema) while producers are still emitting old data, the new consumer can read everything on the topic — the old messages and any still being written by old producers. You upgrade all consumers first, confirm they are happy reading the existing stream, and *then* roll the producers to the new format at your leisure. Consumers lead, producers follow. This is why backward is the sensible default for most teams: the painful side to upgrade is usually the fleet of consumers, and backward lets you get them onto the new schema before a single new-format message exists.

### Why forward means producers first

Forward compatibility guarantees the old schema reads new data. So you can deploy the new producer (emitting the new format) while consumers are still running old code, and those old consumers can read the new messages with their old schema. Producers lead, consumers follow. You choose forward when the producer is the thing you need to change *now* — say it must start emitting a new field for a regulatory deadline — and you cannot wait for every downstream consumer to redeploy first. The new field flows immediately; old consumers ignore it; consumers upgrade whenever they get around to it.

### Why full means any order

Full compatibility guarantees both directions, so there is no ordering constraint at all. Any producer version can talk to any consumer version, and you can roll them out in whatever sequence your CI/CD happens to produce. This is the mode you want when many teams deploy independently and no human is coordinating the sequence, because the only safe assumption in that world is that *every* ordering will happen eventually. The figure below shows the safe-change and breaking-change patterns side by side, which is the visual you want in your head when you are about to choose a mode: the left is the additive change that any mode tolerates, the right is the required-field removal that breaks readers.

![A before and after comparison showing a safe additive change adding an optional field with a default that both old and new readers handle, versus a breaking change removing a required field that strands old readers with a decode error](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-6.webp)

#### Worked example: adding an optional field with a default, old and new consumers both working

Let us prove backward and forward compatibility with a real evolution. Schema v1 of the `orders-value` subject has `order_id`, `customer_id`, `amount_cents`, and `status`. We want to add a `discount_cents` field. We add it **with a default of 0**:

```json
{
  "type": "record",
  "name": "Order",
  "namespace": "com.shop.events",
  "fields": [
    {"name": "order_id",      "type": "string"},
    {"name": "customer_id",   "type": "string"},
    {"name": "amount_cents",  "type": "long"},
    {"name": "status",        "type": "string"},
    {"name": "discount_cents","type": "long", "default": 0}
  ]
}
```

Now walk the four reader-writer combinations that can occur during the rollout. **Case A — new consumer reads an old (v1) message.** The v1 message has no `discount_cents` on the wire. The new consumer's reader schema says "if `discount_cents` is absent, use the default 0." It reads the order with `discount_cents = 0`. No error. This is *backward* compatibility in action: new schema reading old data. **Case B — old consumer reads a new (v2) message.** The v2 message carries `discount_cents` in its bytes. The old consumer's reader schema has never heard of `discount_cents`, so during schema resolution it simply drops the field it does not want. It reads the order with the four fields it knows about and ignores the fifth. No error. This is *forward* compatibility in action: old schema reading new data.

Because both Case A and Case B succeed, this change is **fully compatible**, and the registry will accept it under any mode. The operational payoff is concrete: you can deploy the new producer and the new consumers in any order, in any mix, during the rollout window, and every message — old or new, read by old code or new — deserializes correctly. A field added with a default is the safest change in the entire schema-evolution vocabulary, and it is fully compatible precisely because it survives both reader-writer directions. Compare that to **Case C — what if we had added `discount_cents` with no default.** Then Case A breaks: a new consumer reading an old message finds the field absent with no fallback, and deserialization fails. That single missing `"default": 0` is the line between a clean rollout and a 3am page.

## 7. Evolution rules: safe and unsafe changes

We can now state the full menu of changes and their compatibility impact, which is the reference you will actually keep open while reviewing a schema PR. The matrix below crosses the change types against their backward and forward impact and the verdict; the prose after it explains the *why* behind each verdict, because a rule you understand is a rule you will not violate by accident.

![A matrix crossing change types of add, remove, rename, and retype a field against backward and forward compatibility showing that adding with a default is universally safe while the others break a direction or both](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-9.webp)

### The safe changes

**Add a field with a default.** This is the workhorse safe change, and we just proved it is fully compatible. The default is doing the load-bearing work: it gives a new reader something to use when old data lacks the field, and it lets the field be omittable so old readers ignoring it is harmless. *Never* add a field to a long-lived event without a default unless you have deliberately chosen forward-only compatibility and understand the consequence. **Remove a field that had a default.** This is the inverse and it is also safe in both directions: new readers do not miss a field they removed, and old readers encountering data without it fall back on their own default. The asymmetry to remember is that removing a field is only safe if that field *had a default to begin with*, which is one more reason to give every field a default at birth — you are buying the right to remove it cleanly later.

### The unsafe changes

**Add a required field with no default.** This breaks backward compatibility: a new reader meets old data missing the field and has no fallback, so it fails. It is forward-safe (old readers ignore the new field), which is exactly the trap — it passes a forward-compatibility check and then breaks the instant a new consumer reads old data. If you genuinely need a new field to be required, you cannot get there by evolution; you make it optional-with-default now and enforce required-ness in application logic, or you cut a new schema/topic entirely. **Remove a required field.** The mirror failure: breaks forward compatibility, because old readers that require the field meet new data without it. Removing any field a consumer depends on is a breaking change no matter how you slice it; the registry under full or forward-transitive will stop you, which is the point.

**Rename a field.** This is the change that looks free and is not. To every binary format and the registry, a rename is a remove-plus-add: the old name disappears (breaking anyone keyed on it) and a new name appears (which old readers ignore, so the data is silently lost for them). A rename breaks compatibility in *both* directions and there is no default that saves you. The escape hatch in Avro is **aliases**: you declare `"aliases": ["old_name"]` on the renamed field, and Avro's schema resolution maps the old name to the new one during read, making the rename safe. Protobuf sidesteps the problem entirely because it never put the name on the wire — it carried the field *number* — so a rename in a `.proto` is a no-op as long as the number is unchanged. This is one of Protobuf's genuine ergonomic wins over Avro. **Change a field's type.** Almost always fatal and almost never worth attempting. The bytes for a `long` and the bytes for a `string` are not interchangeable; there is no default that bridges them; both directions break. There are a few narrow promotions Avro allows (int to long, float to double, and reading certain types as their wider cousins), but the safe pattern for a real type change is to add a new field with the new type, dual-write both for a migration window, and retire the old field once all consumers have moved. Never change a type in place on a live subject.

```bash
# Set the compatibility mode for a subject before you evolve it.
# FULL means every change must read both old->new and new->old.
curl -X PUT http://schema-registry:8081/config/orders-value \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  -d '{"compatibility": "FULL_TRANSITIVE"}'

# Dry-run a proposed schema against the mode BEFORE you register it.
# Returns {"is_compatible": true|false} — wire this into CI.
curl -X POST http://schema-registry:8081/compatibility/subjects/orders-value/versions/latest \
  -H "Content-Type: application/vnd.schemaregistry.v1+json" \
  -d @order-v2.avsc.json
```

That `/compatibility` endpoint is the single most valuable thing the registry exposes, because it lets you fail the build instead of failing production. Put it in CI and a breaking schema change becomes a red check on a pull request, reviewed in daylight, instead of an incident at 3am.

## 8. Breaking changes and how to roll them out

Sometimes you genuinely must make a breaking change. The contract was wrong, the type was a mistake, a field has to go. The registry will not let you sneak it past full compatibility, and that is correct — but it does not mean breaking changes are impossible, only that you cannot do them silently in place. There is a disciplined way to roll out a breaking change, and it always involves running two versions side by side for a window long enough that everyone migrates. The timeline figure shows the healthy shape of evolution — additive versions advancing while every consumer keeps decoding — which is the baseline you deviate from only deliberately when a true break is required.

![A timeline showing a schema advancing across versions one through three with additive optional fields while a proposed required-field removal is rejected and all consumers keep decoding under backward compatibility](/imgs/blogs/schema-management-evolution-avro-protobuf-registry-8.webp)

### Pattern one: the parallel-field migration

The cleanest breaking change that is not really breaking. Suppose `amount` was mistakenly a string and must become a numeric `amount_cents`. You do *not* change the type of `amount` in place — that breaks everyone. Instead you **add** `amount_cents` as a new optional field with a default, and for a migration window the producer **dual-writes**: it populates both the old string `amount` and the new `amount_cents`. Consumers migrate one at a time from reading `amount` to reading `amount_cents`, on their own schedule, because both fields are present. Once telemetry confirms no consumer reads `amount` anymore — and you *measure* this, you do not guess — you remove `amount` (safe, because it had a default and nobody reads it). The breaking type change became two fully-compatible additive changes separated by a measured migration window. This is the pattern for the vast majority of "we have to change the type" situations, and it is worth the extra ceremony every time.

### Pattern two: the new topic (a versioned channel)

When the change is too large to dual-write — the whole record is being restructured, the semantics are changing, the event means something different now — do not try to evolve the subject at all. Create a **new topic**: `orders.v2`. The old producer keeps writing `orders`, the new producer writes `orders.v2`, and consumers migrate from one topic to the other when ready. For a window you may run a small bridge that reads `orders` and republishes a converted form onto `orders.v2` so late consumers are not starved. This is the heavier hammer, and you reach for it when the contract is not evolving but being *replaced*. The cost is operational — two topics, a bridge, eventual cleanup — but it cleanly separates the old contract from the new one with no compatibility gymnastics. Treat the topic name as the major version and schema evolution as the minor version: additive changes evolve the subject, breaking changes cut a new topic.

### Pattern three: coordinated big-bang (rarely, and only small)

Occasionally — a tiny internal topic, a single producer and single consumer owned by one team, no replay window — you can take the broker path of least resistance: drain the topic, stop the producer, deploy both sides on the new incompatible schema, restart. This works *only* when you control both ends, the topic is short-retention, and you can tolerate a brief stop. It does not scale past one team and one topic, and the moment a second consumer appears it becomes a footgun. I mention it for completeness and to say: if you find yourself doing big-bang migrations on shared topics, that is the smell that tells you to set up full compatibility and the parallel-field pattern instead.

### The deletion subtlety

Even "remove a field" deserves care across a replayable log. Under non-transitive backward compatibility, removing a field with a default is fine against the latest version — but if you replay from a v1 message that *did* have the field as required-for-the-reader, an old consumer can still break. This is precisely why transitive modes exist and why, on any topic with long retention or replay, you should be on a `_TRANSITIVE` mode. The general lesson: in a log, the past is not gone. Every message you ever wrote can be read again, so every schema you ever registered is potentially live. Compatibility on a log is a property of the *whole history*, not the latest pair.

## 9. Operating a registry in production

A schema registry is a stateful service on the critical path for understanding data, and like any such service it has operational characteristics you must respect or it will surprise you. Here is what actually matters when you run one.

### Availability and the caching that saves you

The registry sounds terrifying at first — "a central service every producer and consumer depends on" — but the caching model defangs it. Clients cache schema-ID-to-schema mappings aggressively and effectively forever (schemas are immutable once registered, so the cache never goes stale). In steady state, a consumer that has seen schema ID 42 once never asks the registry about it again. This means the registry being briefly *unavailable* does not stop your already-running producers and consumers from processing known schemas — they serve from cache. The registry is only on the hot path for *new* schemas: a producer registering a brand-new schema, or a consumer encountering a schema ID it has never cached. So a registry outage degrades gracefully: existing traffic flows, only first-time-schema operations block. You still want it highly available — run it with multiple replicas — but a 30-second blip is not an outage of your data plane. Where Confluent's registry stores its state is itself a Kafka topic (`_schemas`), a single-partition compacted log that is the source of truth, which means the registry's durability inherits Kafka's replication — a tidy bit of dogfooding worth knowing when you reason about its failure modes, and a direct application of the log-as-database idea from the [Kafka deep dive](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage).

### Governance: who can register, and the CI gate

The registry's compatibility check is only as good as your discipline about *when* it runs. The failure mode is letting producers auto-register schemas at runtime: a producer ships, registers a new schema on first publish, the registry checks compatibility *then*, and if it is incompatible the producer crashes in production. That is the check running too late. The fix is to **disable auto-registration in production** (`auto.register.schemas=false`) and instead register schemas explicitly through a CI pipeline that runs the `/compatibility` dry-run check as a required status on every pull request. Now a breaking change is a red CI check on a PR, caught by the author in daylight, not a crash loop on deploy. Pair this with **access control** — only the CI service account can write to the registry — so no one can hand-register a schema that bypassed the check. The registry gives you the mechanism; CI gives you the *enforcement point*, and the enforcement point must be before merge, not at runtime.

```yaml
# A CI step that fails the build on an incompatible schema change.
# Run this on every PR that touches a .avsc file.
schema-compat-check:
  script:
    - |
      for schema in schemas/*.avsc; do
        subject="$(basename "$schema" .avsc)-value"
        result=$(curl -s -X POST \
          "$REGISTRY_URL/compatibility/subjects/$subject/versions/latest" \
          -H "Content-Type: application/vnd.schemaregistry.v1+json" \
          --data-binary "@$schema")
        echo "$subject -> $result"
        # Fail the job if the registry says it is not compatible.
        echo "$result" | grep -q '"is_compatible":true' || exit 1
      done
```

### Subject naming strategy, the choice you make once

The default `TopicNameStrategy` scopes a subject to a topic (`orders-value`), which means a topic carries exactly one evolving schema lineage. That is the right default for most teams. But there are two alternatives worth knowing: `RecordNameStrategy` scopes the subject to the record's fully-qualified name regardless of topic, which lets you put multiple event *types* on one topic (useful for an event-sourcing stream where `OrderPlaced`, `OrderShipped`, and `OrderCancelled` all live on `orders`); and `TopicRecordNameStrategy` combines both. Choose deliberately, because changing it later is painful — it changes the subjects under which schemas are registered, which can orphan history. If you might put multiple event types on one topic (and event-sourced systems often do), pick a record-name strategy up front.

### Metrics, retention, and the things that page you

Watch three signals. **Registry request rate and latency** — a sudden spike in registry calls usually means a client lost its cache (a fleet-wide restart) and is re-fetching, or someone enabled auto-registration and is hammering it; either way it tells you something changed. **Schema count growth** — a subject accumulating versions fast is a smell of churn or of someone fighting the compatibility check by registering variant after variant. **Compatibility-check failures in CI** — track these, because a rising rate means a team is repeatedly proposing breaking changes, which is a signal to go talk to them about the contract before they find a way around the gate. And remember that schemas are *forever*: the registry never deletes a schema that a message somewhere might still reference, so your retention policy on the topic and your transitive-compatibility mode together define how far back into schema history you must remain compatible. A 90-day topic on full-transitive means every schema from the last 90 days must remain mutually readable, and you should be able to state that property out loud when someone proposes a change.

### Bootstrapping a registry into an existing JSON pipeline

Most teams do not get to start with a registry; they inherit a topic that has been carrying JSON for years and need to migrate without a flag day. The pragmatic path is incremental and runs both formats in parallel during the transition. First, write down the *implicit* schema the JSON has been carrying — sample a few thousand live messages, infer the field set and types, and register that as v1 of the subject so the contract finally exists on paper. Second, stand up the registry and switch new producers to the Avro serializer while leaving consumers reading JSON, with a small transformer that reads the Avro topic and republishes JSON onto a compatibility topic for laggards — or, more cleanly, dual-publish to two topics for a window. Third, migrate consumers one at a time onto the registry-aware deserializer, watching that each one decodes correctly against real traffic before moving the next. Finally, when telemetry shows no consumer reads the JSON path, retire it. The whole migration is a sequence of the same parallel-running discipline you use for any breaking change, and the key is to *measure* the cutover rather than assume it — the registry's metrics tell you exactly when the last JSON reader went away.

### Multi-datacenter and the federation question

In a multi-region deployment, you need the same schema IDs to mean the same schemas everywhere, or a message produced in one region cannot be decoded by a consumer that fetched a conflicting schema in another. The standard answer is a single primary registry that owns ID assignment, replicated read-only to other regions, so IDs are globally consistent and only the primary issues new ones. Getting this wrong — two registries independently assigning ID 42 to different schemas — produces the worst kind of bug, where a message decodes into the wrong shape silently in one region and correctly in another. If you operate across regions, make ID assignment single-sourced and treat the schema-replication lag as a real dependency in your cross-region message flow.

## Case studies and war stories

### The Tuesday-afternoon field that paged three teams

This is the opening story, and it is composited from a pattern I have watched play out at more than one company. An upstream team added an innocuous field to a high-fanout event with **no schema governance** — JSON on the topic, no registry, no compatibility check. The producer's tests passed because the producer does not deserialize its own output. Downstream, a consumer that did strict JSON-schema validation rejected every message with the unexpected field; a second consumer that parsed positionally read the wrong values; the warehouse loader, configured for an exact column set, dropped rows. Three teams paged, none of whom made the change. The lesson is the thesis of this whole post: **the breaker was safe and the broken were paged**, and the only fix is to move the contract check *upstream of the producer's deploy* — a registry with a compatibility gate in CI would have turned all three pages into one red check on the original PR. They adopted Avro and a registry the following quarter; the class of incident disappeared.

### The non-transitive replay that broke a year later

A team ran a registry on the default non-transitive backward compatibility and evolved a subject through a dozen versions over a year, each change backward-compatible with the *previous* version. Then they ran a full reprocessing job — replaying the topic from the beginning to rebuild a derived store. The job crashed on messages from eight months prior, because v12's schema was not backward-compatible with v4's data even though it was compatible with v11's. The chain of pairwise-compatible steps had drifted, exactly as non-transitive allows. The fix was conceptual and then mechanical: recognize that **a replayable log makes the entire history live**, switch the subject to `BACKWARD_TRANSITIVE`, and accept that some future changes would be stricter. The deeper lesson is that compatibility mode must match retention reality — long retention or replay means transitive, full stop.

### The Protobuf number reuse that corrupted data silently

A team using Protobuf removed a deprecated field, `region = 7`, from a `.proto`. Months later a different engineer added a new field and, seeing that `7` was "free," reused it: `tier = 7`. Old messages on the topic still carried a `region` value under number 7. New consumers, reading those old messages with the new `.proto`, decoded the old `region` bytes as a `tier` — no error, no crash, just *wrong data* flowing through the system, silently, for weeks. This is the Protobuf failure mode that the format's docs warn about in bold: **never reuse a field number.** The fix is the `reserved` keyword — `reserved 7;` — which permanently blocks the number (and the name) from ever being reassigned. The broader lesson is that binary-with-numbers gives you rename-safety for free but hands you a new sharp edge in number management, and the registry's compatibility check does *not* catch number reuse on its own unless you treat it correctly, so discipline plus `reserved` is mandatory.

### The auto-registration crash loop

A startup left `auto.register.schemas=true` (the default in some client versions) and let producers register schemas at runtime. A developer pushed a schema change that was backward-incompatible, the producer deployed, tried to register on first publish, the registry rejected it, and the producer entered a crash loop — failing to start, restarting, failing again — taking the producing service down entirely. The change had passed code review because nobody runs the producer's serialization in review. The fix was the standard production posture: **turn off auto-registration**, register through CI with the dry-run compatibility check as a required gate, and lock registry writes to the CI service account. The incompatible change then surfaces as a failed CI job on the PR, which is exactly where you want to discover it, instead of as a production crash loop discovered by an alert.

## When to reach for this (and when not to)

Reach for a **schema registry and a binary format** the moment more than one team consumes a topic, or the moment a topic carries meaningful volume, or the moment you have a replay window measured in days. Those three conditions — multiple consumers, real volume, replayability — are each independently sufficient, and most production topics hit all three. With multiple consumers you need the governance because the contract spans teams; with volume you need the size win; with replay you need transitive compatibility because the history is live. In that world, Avro-or-Protobuf plus a registry plus a CI compatibility gate is not gold-plating, it is the baseline that keeps the system operable.

Do **not** reach for it when the honest answer is that you have a single producer and single consumer owned by one team, a short-retention internal topic, low volume, and a debugging workflow that benefits from eyeballing messages. There, JSON with a checked-in JSON Schema (or even just discipline) is fine, and a registry is operational overhead you do not need yet — one more stateful service to run, monitor, and keep available. The trap is staying on JSON *past* the point where a second team starts consuming, because that transition is silent: nothing forces you to adopt governance, right up until the day an ungoverned change pages the new consumer. The decision rule: introduce the registry the day the *second* independent consumer appears, before it breaks, not after. And once you have it, default new shared subjects to `FULL_TRANSITIVE` so that any team can deploy in any order against the full history without coordination — pay the strictness up front to buy the operational freedom forever.

Between Avro and Protobuf specifically: choose Protobuf if you already run gRPC and value field-number stability and rename-safety; choose Avro if you live in the Kafka and Confluent data ecosystem and want the strongest reader-default resolution semantics and the tightest registry integration. Both are correct; the wrong choice is JSON-at-scale-without-a-schema, which is the one decision that reliably generates the 3am page.

## Key takeaways

- **The schema is the API between a producer and a consumer.** Decoupling in time and deployment does not decouple meaning; make the contract an explicit, versioned, machine-checked artifact or it is owned by nobody until it breaks.
- **There are four ways a contract breaks** — missing field, extra field, type change, rename — and every evolution rule is a policy about which is allowed. A rename is a remove-plus-add and breaks both directions.
- **Binary-with-schema beats JSON on size and CPU.** At 100k msg/s an 800-byte JSON record versus a 75-byte Avro record saves roughly 72.5 MB/s on the wire and over 6 TB a day before replication — pure overhead reclaimed.
- **A registry ships IDs, not schemas.** Producer registers and gets an ID; each message carries a 5-byte magic-plus-ID header; the consumer resolves the ID to a schema once and caches it forever. The full schema crosses the network twice, not per message.
- **Backward = consumers first, forward = producers first, full = any order.** Derive it from "which side reads whose data," and default shared subjects to full so independent teams deploy in any sequence.
- **Use transitive modes on any replayable or long-retention topic.** Non-transitive checks only the latest pair; a log makes the entire history live, and pairwise compatibility silently drifts apart over a dozen versions.
- **Add fields with a default; never remove a required field, change a type in place, or rename without an alias.** The default is the load-bearing element of every safe change; Avro aliases and Protobuf field numbers are the rename escape hatches.
- **Roll out true breaking changes with parallel fields or a new topic.** Dual-write old and new, migrate consumers on measured telemetry, then retire the old field — never big-bang a shared topic.
- **Operate the registry with auto-registration off and a CI compatibility gate on.** Catch breaking changes as a red check on a pull request in daylight, not as a production crash loop at 3am; lock registry writes to the CI service account.
- **Never reuse a Protobuf field number; use `reserved`.** Number reuse corrupts old data into the wrong field silently, and the compatibility check will not save you from it.

## Further reading

- [Anatomy of a message system: producers, brokers, consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) — where serialization was established as a producer responsibility and the format as a contract with every consumer.
- [Kafka deep dive: log segments, page cache, storage](/blog/software-development/message-queue/kafka-deep-dive-log-segments-page-cache-storage) — why a smaller payload directly improves page-cache hit rate and sequential-read throughput, and how the registry stores its own state as a compacted log.
- [Event sourcing and CQRS on a commit log](/blog/software-development/message-queue/event-sourcing-and-cqrs-with-an-event-log) — the canonical case for transitive compatibility, where the entire history is replayable and every schema you ever registered stays live.
- [Confluent Schema Registry documentation](https://docs.confluent.io/platform/current/schema-registry/index.html) — the canonical implementation: subjects, compatibility modes, the REST API, and the wire format.
- [Apache Avro specification](https://avro.apache.org/docs/current/specification/) — the schema-resolution rules, defaults, aliases, and the precise semantics of reader-versus-writer schemas.
- [Protocol Buffers language guide](https://protobuf.dev/programming-guides/proto3/) — field numbers, the `reserved` keyword, and the rules for evolving a `.proto` without breaking the wire format.
