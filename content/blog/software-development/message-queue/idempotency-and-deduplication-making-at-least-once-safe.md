---
title: "Idempotency and Deduplication: Making At-Least-Once Delivery Safe"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Learn why at-least-once delivery makes duplicates a steady-state condition rather than a rare bug, and master the discipline that makes it safe: idempotency keys, dedup tables, conditional writes, the inbox pattern, and effectively-once across external side effects."
tags:
  [
    "message-queue",
    "idempotency",
    "deduplication",
    "at-least-once",
    "effectively-once",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "reliability",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 53
image: "/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-1.webp"
---

A few years ago I watched a payments team spend a frantic Tuesday refunding two thousand customers who had each been charged twice for the same order. Nothing was broken. The database was healthy, the queue was healthy, the consumer code passed every test. What happened was the most ordinary event in distributed systems: a consumer processed a message, called the payment provider, the charge succeeded, and then — a few hundred milliseconds before the acknowledgement could be written — the consumer pod was killed by a routine deployment. The broker, having never seen an ack, did exactly what at-least-once delivery promises: it redelivered the message. The replacement pod picked it up, called the provider again, and charged the card a second time. The system did everything right. The *application* had simply never been told that "process this message" might mean "process this message again," and so it had no way to recognize the second delivery as a repeat of the first.

That is the entire subject of this post. If you have read the [delivery semantics](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) post in this series, you already know the punchline of the layer below: at-least-once is the realistic default, exactly-once delivery is a category error at the wire, and duplicates are not an anomaly to be patched but a guaranteed consequence of a lost acknowledgement on a healthy network. This post is about the layer above — the discipline that takes that messy reality and makes it *safe*. That discipline is idempotency, supported by deduplication, and it is the single most important application-level skill in event-driven systems. The figure below is the whole argument in miniature: the same duplicate delivery, run through a non-idempotent consumer, charges the card twice; run through an idempotent one with a dedup key, it charges once and the duplicate becomes a harmless no-op.

![A two-column before and after diagram contrasting a non-idempotent consumer that charges a card twice against an idempotent consumer that uses a dedup key to bill exactly once](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-1.webp)

By the end of this post you will be able to define idempotency precisely enough to argue about it, recognize which of your operations are naturally idempotent and which must be made so, choose a deduplication strategy with eyes open about its write cost and correctness, size a dedup window against real throughput numbers, decide where in your pipeline to dedup and why the consumer is almost always the honest place, and handle the genuinely hard case where the side effect leaves your system entirely — a card charge, an email, a webhook. We will end at the inbox pattern and the precise meaning of "effectively-once," which is the only form of exactly-once you can actually ship. Two sibling posts go deeper on adjacent machinery: the [transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) for making your *publishes* reliable and idempotent, and [exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) for how transactions wire the producer and consumer sides together. This post owns the consumer-side machinery they both lean on.

## 1. Why at-least-once forces you to think about duplicates

Let us be unambiguous about why duplicates are unavoidable, because the whole rest of the post is a response to this one fact. A message queue sits between a producer and a consumer, and the consumer's job is to do some work and then tell the broker "done, you can stop tracking this one." That acknowledgement travels back across a network. The network can lose the acknowledgement, delay it past a timeout, or deliver it after the broker has already given up waiting. The broker cannot distinguish a consumer that crashed before doing the work from a consumer that did the work and then crashed before the ack arrived from a consumer that did the work and acked successfully but had the ack eaten by a transient network blip. From the broker's vantage point all three look identical: no ack arrived in time. It has exactly two options. It can assume the work was done and move on — that is at-most-once, and if it guessed wrong the message is lost forever. Or it can assume the work was not done and redeliver — that is at-least-once, and if it guessed wrong the work runs a second time.

Every serious queue defaults to at-least-once, because losing a message silently is almost always worse than doing harmless extra work — *provided* the extra work is in fact harmless, which is precisely the property this post teaches you to guarantee. Kafka with its default consumer commit behavior, RabbitMQ with manual acks, Amazon SQS standard queues, Google Pub/Sub, NATS JetStream — all of them, in their normal and recommended configurations, deliver at-least-once. The duplicate is not a bug in any of them. It is the contract. When you read "at-least-once delivery" in a broker's documentation, the correct mental translation is: *this system will redeliver your message whenever it is unsure, and it is unsure constantly, so build for repeats.*

How often is "constantly"? More often than most engineers guess. Acknowledgement loss is not the only source of duplicates. Consider the menagerie of ordinary events that each produce a redelivery: a consumer that exceeds its visibility timeout or `max.poll.interval.ms` because one message took too long to process, so the broker concludes it is dead and hands the message to someone else; a consumer group rebalance in Kafka, where a partition is revoked from one consumer and assigned to another after the first had already processed but not yet committed an offset; a producer retry, where the producer sent a message, got no ack, and resent it, so the *same logical event* now exists twice in the log before any consumer has touched it; a broker failover, where an in-flight, un-replicated acknowledgement is lost when leadership moves to a follower; an application-level retry in your own code, where you catch an exception, log it, and re-enqueue. Each of these is a normal Tuesday. Stack them across a fleet of hundreds of consumers running millions of messages a day and duplicates stop being rare and become a steady drizzle — a small but never-zero fraction of all deliveries.

### Duplicates are a rate, not an event

The most useful reframing I can offer is to stop thinking of a duplicate as an incident and start thinking of it as a rate. In a healthy production system the duplicate rate might be one in ten thousand deliveries, or one in a hundred thousand, depending on your rebalance frequency, deploy cadence, and timeout margins. It is small. But "small and constant" is a completely different engineering posture from "rare and exceptional." A rare-and-exceptional bug you handle with an alert and a runbook. A small-and-constant rate you handle with a *mechanism* that runs on every single message, because you cannot know in advance which one in ten thousand is the repeat. That mechanism is deduplication, and the property it preserves is idempotency. The cost discipline of the post is that this mechanism runs on the other 9,999 messages too — the ones that are not duplicates — so it had better be cheap.

There is a tempting wrong turn here, which is to try to make the duplicates go away at the delivery layer — to demand "exactly-once delivery" from the broker and wash your hands of the problem. The delivery semantics post dismantles that hope in full, so I will only summarize: you cannot make message delivery unique across an unreliable network, because the network cannot distinguish a lost message from a lost acknowledgement, and any protocol that tries reduces to the two-generals problem, which is provably unsolvable. What you *can* do is make the message's *effect* land once even when the message is delivered many times. The duplicates still happen on the wire; you just make them not matter. That is the pivot from "deliver once" to "apply once," and idempotency is the name of the property that lets you apply once.

## 2. What idempotency actually means

Here is the definition, and it is worth memorizing in exactly these words: **an operation is idempotent if applying it twice produces the same result as applying it once.** More generally, applying it any number of times beyond the first produces no additional effect. The word comes from mathematics — a function `f` is idempotent if `f(f(x)) = f(x)` — and the mathematical precision is the point. Idempotency is not "the operation is safe to retry." It is the stronger and more specific claim that the *state of the world after N applications is indistinguishable from the state after one application.*

Let me draw the line sharply, because this is where most confusion lives. Consider three operations against a bank balance. "Set the balance to 100" is idempotent: run it once and the balance is 100; run it five more times and the balance is still 100. The repeats are invisible. "Add 50 to the balance" is *not* idempotent: run it once and you add 50; run it again and you add another 50; the result depends on how many times it ran, which is exactly the property we cannot control under at-least-once. "Append a row to the transactions table" is also not idempotent: each append adds a row, so two deliveries leave two rows where there should be one. The pattern is clear and it is the most important pattern in this whole subject: *operations that compute the new state purely from the inputs (absolute assignments, upserts to a known value) are idempotent; operations that compute the new state by reading the old state and modifying it (increments, appends, decrements, "add one more") are not.*

A subtle but critical refinement: idempotency is a property of an operation *with respect to a particular notion of "same."* When we say two applications produce "the same result," we have to be precise about which observable result we mean. The HTTP spec, for example, defines idempotency in terms of the *server's state*, not the *response the client sees*. A `DELETE /orders/42` is idempotent because after the first call the order is gone and after the second call the order is still gone — the server state is identical. But the first call returns `200 OK` and the second returns `404 Not Found`. The responses differ; the state does not. For our purposes — making side effects safe under redelivery — the notion of "same" that matters is the externally observable state: the customer's balance, the number of emails they received, the count of rows in a table, the existence of a charge on a card. We do not care whether the second delivery's *response* matches the first; we care that the second delivery does not *add a second charge*.

### Why this is harder than it looks

If idempotency were just "use SET instead of ADD," this would be a short post. The difficulty is that real business operations are rarely a single naturally-idempotent write. "Process this order" might mean: decrement inventory by the ordered quantity, charge the customer's card, insert a fulfillment record, increment a daily-orders metric, and publish three downstream events. Decrementing inventory is a non-idempotent read-modify-write. Charging the card is a non-idempotent external side effect. Incrementing a metric is non-idempotent. Publishing events is non-idempotent. *None* of the natural building blocks is idempotent, and the operation as a whole is a composite of five non-idempotent steps that must, together, behave as if they ran once even when the message is delivered twice. Making *that* idempotent is real engineering, and it is what the dedup-key and inbox machinery in the later sections is for. The natural-idempotency section that follows is the foundation; the keyed-dedup sections are how you reach idempotency when nature does not hand it to you for free.

One more framing before we go deeper. The reason idempotency is the right tool — rather than, say, "make sure duplicates never happen" — is that idempotency is a *local* property you can establish at the one place that has the side effect, without coordinating with the broker, the network, or every other consumer. You do not need a global agreement that this message has been seen. You need your own write to be shaped so that a repeat is invisible. Local properties are cheap and robust; global agreements are expensive and fragile. The entire engineering strategy of this post is: push the safety down to the local write, and never try to make the distributed delivery unique.

## 3. Natural idempotency: SET vs INCREMENT

The cheapest idempotency is the kind you get for free because the operation is already shaped right. Before you reach for a dedup table or a key store, always ask: can I reformulate this operation so that it is naturally idempotent? Surprisingly often the answer is yes, and the reformulation costs nothing at runtime — no extra read, no extra store, no extra latency. This is the first lever you should pull, every time, before anything more elaborate.

The contrast at the heart of natural idempotency is SET versus INCREMENT, and the figure below makes it concrete with a balance. An INCREMENT is relative: it reads the current value and adds to it, so two deliveries compound into double the intended change. A SET is absolute: it overwrites with a value computed from the message alone, so two deliveries land on the same final value and the second is invisible. The same logic generalizes far beyond numeric balances. "Set the order status to SHIPPED" is idempotent; "advance the order to the next status" is not. "Set the user's last-seen timestamp to the event time" is idempotent; "bump the user's login count" is not. "Upsert this product record with these fields" is idempotent; "insert this product record" fails on the second delivery with a duplicate-key error or, worse, creates a second row if there is no constraint.

![A two-column before and after diagram showing that an INCREMENT operation compounds to the wrong total under redelivery while a SET upsert converges to the correct value](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-7.webp)

### Reformulating increments into sets

The most valuable trick in this section is converting a non-idempotent increment into an idempotent set by carrying the *absolute target* in the message rather than the *delta*. Imagine you are consuming events that update a running inventory count. The naive event says "inventory changed by -3." That is an increment, and a duplicate double-counts the decrement. The fix is to make the producer compute and send the *resulting absolute value*: "inventory is now 47." Now the consumer does a SET, and a duplicate delivery sets it to 47 a second time, which is a no-op. The producer had to know the absolute value, which it usually does because it is the system of record. You have moved the non-idempotent read-modify-write from the consumer — where redelivery is uncontrolled — into the producer, where it can be done once under a lock or a transaction.

This does not always work, and it is honest to say when it does not. If multiple independent producers can change the same value concurrently, then "inventory is now 47" from producer A and "inventory is now 52" from producer B race, and last-writer-wins silently loses one update. Absolute SETs are idempotent but they are *not* commutative or concurrency-safe by themselves. When you have concurrent writers you need either a single-writer-per-key discipline (partition the key so only one consumer ever touches it — which Kafka's key-based partitioning gives you naturally) or a version-stamped conditional write, which we cover in the conditional-write section. Natural idempotency via SET is the first and cheapest tool, but it is a tool with a clear edge: it is safe under redelivery, not necessarily under concurrency.

There is a second, subtler limit worth naming: a SET is idempotent only when the value you set does not itself depend on the order of messages. If two events both SET a field but to *different* correct values — "status is SHIPPED" then "status is DELIVERED" — then redelivery of the older SHIPPED message *after* the DELIVERED one has already landed will incorrectly revert the order to SHIPPED. The SET is idempotent with respect to a single message (applying it twice is the same as once) but it is not safe against *out-of-order* redelivery of *different* messages. This is precisely where conditional writes earn their place: gating the SET on the expected prior state or a monotonic version makes it reject the stale revert, turning order-sensitivity into a guard the database enforces. So the honest layering is: SET for single-message redelivery safety, conditional write when message order can be scrambled, and partitioning when you need a single writer per key. Each tool handles a distinct failure mode, and real pipelines often need two of them together.

Here is the reformulation in code, deliberately boring because boring is the point:

```python
# NON-idempotent: a redelivered message double-decrements
def handle_inventory_event_bad(event):
    db.execute(
        "UPDATE inventory SET qty = qty + %s WHERE sku = %s",
        (event["delta"], event["sku"]),  # delta = -3, applied twice = -6
    )

# Idempotent: the producer sends the absolute resulting value
def handle_inventory_event_good(event):
    db.execute(
        "UPDATE inventory SET qty = %s WHERE sku = %s",
        (event["new_qty"], event["sku"]),  # new_qty = 47, set twice = 47
    )
```

### Upserts: idempotency for "create or update"

The database-level workhorse of natural idempotency is the upsert — `INSERT ... ON CONFLICT DO UPDATE` in PostgreSQL, `INSERT ... ON DUPLICATE KEY UPDATE` in MySQL, `MERGE` in SQL Server and Oracle. An upsert keyed on a stable business identifier turns "create this entity, or update it if it already exists" into a single idempotent statement. The first delivery inserts the row; the second delivery finds the conflict on the key and updates the row to the same values, which is a no-op in terms of observable state. The crucial ingredient is that the conflict is detected on a *stable key derived from the message*, not on an auto-generated surrogate. If your primary key is a database-assigned auto-increment id, two deliveries produce two rows with two different ids and the upsert never fires. You must have a column — a business key or a message id — that is the same across redeliveries and is constrained to be unique. That observation is the bridge to the next section: idempotency keys.

```sql
-- Idempotent create-or-update keyed on a stable order_id
INSERT INTO orders (order_id, customer_id, status, total_cents)
VALUES ($1, $2, 'CONFIRMED', $3)
ON CONFLICT (order_id) DO UPDATE
  SET status = EXCLUDED.status,
      total_cents = EXCLUDED.total_cents;
-- second delivery: ON CONFLICT fires, sets the same values, no new row
```

## 4. Idempotency keys and dedup keys

When an operation is not naturally idempotent — and most real operations are not — you make it idempotent by attaching a stable identifier to each logical message and using that identifier to recognize repeats. This identifier goes by two closely related names that are worth distinguishing. An **idempotency key** is a token that *you* generate to mark a single logical operation, typically passed to a downstream system so that *it* can deduplicate on your behalf (this is the term payment providers use). A **dedup key** is the identifier *your* consumer uses to recognize that it has already processed a message. They are the same idea applied at two different boundaries, and a well-built consumer often uses both: a dedup key to skip its own repeated processing, and an idempotency key to make the external side effect safe. The figure two sections down shows both in one flow.

The single most important property of either key is *stability*: the same logical message must carry the same key across every redelivery. If the key changes between deliveries, deduplication is impossible — every delivery looks new. This sounds obvious and is violated constantly. The classic mistake is to use a timestamp, a UUID generated at receive time, or the broker's own message offset as the key. None of those is stable across a redelivery. A UUID generated when the consumer receives the message is different on the second delivery. A timestamp differs by milliseconds. Even the broker offset can differ: in Kafka a producer retry can write the same logical event to two different offsets, and after a rebalance the same offset can be re-read, so the offset is neither uniquely-per-event nor stable in the way you need.

### Where a stable key comes from

There are exactly two good sources of a stable dedup key, and you should reach for them in this order.

First, a **business key** intrinsic to the domain event: the order id, the payment id, the `transfer_id`, the `(user_id, event_type, billing_period)` tuple. This is the best key because it encodes what "the same operation" *means* in your domain. Two messages that both say "charge order 8842" are the same operation precisely because they share the order id, and the business key captures that exactly. Business keys also survive things that break technical keys — a message reformatted by an intermediary, republished after a schema migration, or re-emitted by a backfill job will still carry the same order id.

Second, a **producer-assigned message id**: a UUID or content hash the producer stamps onto the message *once, at creation time*, and that travels with the message through every hop and every redelivery. The discipline is that the producer generates it exactly once when the logical event is born, never on a resend. If your producer is a web request handler creating an order, it generates the message id when the order is created, stores it with the order, and stamps it on the event. A producer retry resends the same message id; a downstream republish carries it forward. This is the key to use when there is no natural business key, or when you want to dedup at a level more granular than the business entity (the same order might legitimately produce multiple distinct events, each needing its own id).

A content hash — say SHA-256 of the canonical message body — is a tempting third option and it works when "same content means same operation," but be careful: two genuinely distinct operations that happen to have identical content (two separate \$10 top-ups a user really did make, one minute apart) will hash to the same key and the second will be wrongly suppressed. Content hashing conflates "identical bytes" with "identical operation," and those are not the same thing when legitimate repeats exist. Use a content hash only when you are certain the content uniquely identifies the operation, or combine it with a timestamp window. The general rule: prefer a key that the *domain* says identifies the operation, fall back to a producer-stamped id, and use content hashes only with care.

### Propagating the key

A key is only useful if it survives every hop. The discipline is to put the key somewhere it cannot be lost: a message header, not just the body. In Kafka, stamp it as a record header (`idempotency-key`). In RabbitMQ, use the AMQP `message-id` property, which exists in the protocol precisely for this. In SQS, use a message attribute, or for FIFO queues the built-in `MessageDeduplicationId`. Putting the key in a header rather than burying it in the body means intermediaries, dead-letter handlers, and observability tools can all see and preserve it without parsing your payload, and it means a re-serialization of the body cannot accidentally drop it.

```python
# Producer stamps a stable id ONCE when the logical event is born
import uuid

def create_order(customer_id, items):
    order_id = generate_order_id()            # business key
    msg_id = str(uuid.uuid4())                # stamped once, never on resend
    save_order(order_id, msg_id, items)       # persist the id with the entity
    producer.send(
        topic="orders",
        key=order_id.encode(),                # partition by order_id
        value=serialize(items),
        headers=[("idempotency-key", msg_id.encode())],  # travels every hop
    )
    return order_id
```

## 5. Deduplication strategies (dedup table, unique constraint, upsert, conditional write)

Once every message carries a stable key, deduplication is the act of remembering which keys you have already processed and skipping the repeats. The shape of the dedup flow is the same regardless of strategy, and the figure below captures it: receive the message and extract its key, check the dedup store, either skip (if the key is known) or process (if it is new), and record the key only after the work succeeds. The ordering of "record the key" relative to "do the work" is the crux of correctness, and we will return to it sharply.

![A pipeline diagram showing the deduplication flow from receiving a message and extracting its key, through checking the dedup store, processing or skipping, recording the key with a TTL, and acknowledging](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-2.webp)

There are five strategies worth knowing, and they differ in write cost, in the size of the window they can cover, and in how strong a correctness guarantee they give you. The matrix below lays them out; the prose then walks each one.

![A matrix comparing five deduplication strategies across write cost, window coverage, and correctness strength](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-3.webp)

### Strategy 1: a dedup table or store with TTL

The most general strategy is an explicit store of seen keys — a table in your primary database, a Redis set with per-key expiry, a DynamoDB table with TTL, or a dedicated bloom-filter-plus-store hybrid. On each message you read the store to ask "have I seen this key?", process if not, and write the key with an expiry. The strength of this approach is flexibility: it works for *any* operation, idempotent or not, because the dedup logic is decoupled from the business write. The weakness is correctness around the boundary between the dedup write and the business write. If they are not atomic, you have a window where you can do the work but crash before recording the key (leading to a duplicate on redelivery) or record the key but crash before doing the work (leading to a *lost* message, because the redelivery now thinks it is a duplicate and skips). We address this race directly in section 9 with the inbox pattern, which makes the dedup write and the business write a single transaction. For now, hold the warning: a separate dedup store is correct only if the key-write and the side effect are atomic, or if the side effect is itself idempotent so a missing key-write merely causes a harmless reprocess.

```python
# Dedup table strategy with Redis, TTL = 7 days
def handle(msg):
    key = msg.headers["idempotency-key"]
    # SET NX = set only if not exists; returns False if key already present
    is_new = redis.set(f"dedup:{key}", "1", nx=True, ex=7 * 24 * 3600)
    if not is_new:
        return  # duplicate: skip silently
    process_and_side_effect(msg)  # NOTE: not atomic with the SET above
```

Be honest about the gap in that snippet: the `SET NX` and `process_and_side_effect` are two separate operations. If the process crashes between them, the key is recorded but the work never ran, and you have lost the message. For non-idempotent external side effects this ordering is wrong — you want to record the key *after* the side effect, or make the two atomic, which is exactly the tension that motivates the inbox pattern.

### Strategy 2: a unique constraint in the database

The single most robust dedup mechanism — and my default recommendation whenever the work culminates in a database write — is a unique constraint on the dedup key in the database itself, so the database *enforces* deduplication as part of the same transaction that does the work. You insert a row whose primary key or unique index is the idempotency key, in the same transaction as the business write. The first delivery inserts and commits. The second delivery's insert violates the unique constraint, the whole transaction rolls back, and you catch the constraint violation and treat the message as a successful duplicate. The beauty of this approach is that correctness is *free and atomic*: the database guarantees that exactly one transaction can insert a given key, and because the dedup insert and the business write share a transaction, there is no window where one happens without the other. There is no race to reason about because the database's serializability gives you the mutual exclusion for nothing.

```python
def handle(msg):
    key = msg.headers["idempotency-key"]
    try:
        with db.transaction():
            # dedup insert and business write share ONE transaction
            db.execute("INSERT INTO processed_messages (key) VALUES (%s)", (key,))
            do_business_write(msg)            # atomic with the dedup insert
    except UniqueViolation:
        # second delivery: constraint blocked it, work already done once
        pass
```

The cost is the index maintenance — every dedup key is an index entry, and a high-throughput stream produces a lot of them, which is exactly the storage-cost question of the next section. But for correctness, a unique constraint inside the business transaction is the gold standard, and you should prefer it whenever the work lands in a transactional database.

### Strategy 3: upsert

When the business write is itself a create-or-update keyed on the dedup key, the upsert *is* the dedup, and you need no separate store at all. This is natural idempotency from section 3 applied with the dedup key as the conflict target. The first delivery inserts; the second hits the conflict and overwrites with identical values. The write cost is a single statement — cheaper than the unique-constraint strategy because there is no second row to maintain — and correctness is atomic by construction. The limitation is that it only works when the entire effect of processing the message is captured by that one upserted row. If processing also charges a card or sends an email, the upsert dedups the row but not the external effect, and you are back to needing keys at the external boundary.

### Strategy 4: conditional write (compare-and-set)

A conditional write — compare-and-set — makes a write conditional on the current state matching an expected value, so a redelivery that finds the state already advanced becomes a no-op. The canonical form carries a version or a sentinel in the message and writes only if the stored version is the expected predecessor. "Set status to SHIPPED *where status is CONFIRMED*" advances the order on first delivery; on redelivery the status is already SHIPPED, the `WHERE` matches zero rows, and the update is a no-op. This is idempotency *and* concurrency-safety in one, because it also rejects out-of-order and conflicting writes, not just duplicates. DynamoDB's condition expressions, etcd's compare-and-swap, and SQL's `UPDATE ... WHERE version = N` are all this pattern.

```sql
-- Conditional write: advances only from the expected prior state
UPDATE orders
   SET status = 'SHIPPED', version = version + 1
 WHERE order_id = $1
   AND status = 'CONFIRMED'   -- redelivery finds SHIPPED, matches 0 rows
   AND version = $2;          -- and/or guards against stale writes
```

The strength is that it composes idempotency with optimistic concurrency control, which you often need anyway. The cost is that you must carry the expected prior state in the message or read it first, and you must handle the "zero rows matched" case as a possibly-legitimate duplicate rather than an error.

### Strategy 5: the idempotent-receiver pattern

The idempotent-receiver is the in-memory or fast-cache variant: the consumer keeps a set of recently-seen keys (an LRU cache, a ring buffer, an in-process bloom filter) and drops any message whose key it has seen. It is the cheapest strategy — no database round-trip — and the right one when duplicates cluster in time (a retry storm delivers the same message a few times within seconds) and when missing an occasional duplicate is tolerable. Its weakness is that the memory is *not durable*: a consumer restart loses the seen-set, so a duplicate that arrives after a restart slips through. The idempotent-receiver is a *first line of defense* that catches the common, clustered-in-time duplicates cheaply, layered in front of a durable strategy that catches the rare, spread-out ones. Used alone for a side effect that must never repeat, it is not enough.

## 6. The dedup window and its storage cost

Every dedup strategy that remembers keys must answer one question: *how long do I remember each key?* Remember too short a window and a late duplicate — one that arrives after the key has expired — slips through and causes the exact double-effect you were trying to prevent. Remember forever and your dedup store grows without bound, which at high throughput is a real and expensive problem. The dedup window is the duration you keep a key, and choosing it is a sizing exercise with a clear correctness floor and a clear storage ceiling.

The figure below puts the window on a timeline. The key is recorded at t0 on first delivery. A duplicate at t+5 seconds is caught — typical for a retry storm. A duplicate at t+2 hours is still caught, because the window is wide. At t+7 days the key expires and the storage is reclaimed. The danger zone is everything after expiry: a duplicate that somehow arrives at t+7 days plus one second is no longer recognized and lands a second effect. So the window must be at least as long as the *longest possible gap between a message's first delivery and its last possible redelivery.*

![A timeline showing a dedup key recorded at t0, a duplicate caught at five seconds, another caught at two hours, the key expiring at seven days, and a late duplicate after expiry that is no longer safe](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-6.webp)

### Sizing the window for correctness

What sets the longest redelivery gap? It is the maximum time a message can live in the system before it stops being eligible for redelivery. For a broker with bounded retention and retries, that is roughly: the broker's retention period, plus the maximum retry backoff schedule, plus the time a message can sit in a dead-letter queue before being replayed, plus a safety margin. For Kafka, if your topic retains messages for 7 days and a consumer can legitimately re-read from an old offset within that window after a long outage, your dedup window must cover the full 7 days of retention. For SQS, the maximum message retention is 14 days, so a message could in principle be redelivered up to 14 days after it was first produced, and a dedup window shorter than that has a hole. The rule: *set the dedup window to the maximum age at which a redelivery of the same message is still possible, plus margin.* A common, defensible choice is 7 days for Kafka-backed pipelines and matching the retention for SQS. Shorter windows are tempting for storage reasons but they trade a storage saving for a correctness hole, and the hole is exactly the late, rare duplicate that is hardest to debug because it does not reproduce in testing.

There is one important subtlety: if the *only* duplicates you actually experience come from retry storms and rebalances — which cluster within seconds to minutes — then a window of, say, one hour catches essentially all of them, and the 7-day window only protects against the exotic case of a multi-day-delayed replay. Whether you need the long window depends on whether multi-day-delayed redelivery is possible in your system. If you never replay from old offsets and your retries are bounded to minutes, a short window is genuinely safe and far cheaper. Know your redelivery sources and size to the worst one you actually have, not to a worst case that cannot occur in your topology.

#### Worked example: dedup-window storage at scale

Let us put real numbers on the storage cost, because "remember every key for 7 days" sounds free until you multiply it out. Suppose a stream runs at a sustained **50,000 messages per second**, and you have decided on a **7-day** dedup window to cover full Kafka retention.

First, how many keys must you hold at steady state? In 7 days you receive `50,000 × 86,400 × 7 = 50,000 × 604,800 = 30,240,000,000` messages — about **30.2 billion** keys live in the window at any moment, since you keep each for the full 7 days before it expires.

Now the storage. The per-key cost depends on what you store. A minimal dedup entry is just the key. If the key is a 16-byte UUID stored raw, that is 16 bytes of payload. But no real store holds bare 16-byte values for free — there is index overhead, per-entry metadata, and expiry bookkeeping. In Redis, a single key with a TTL costs on the order of 60–100 bytes once you account for the key string, the value, the expiry, and the dictionary entry overhead; call it 80 bytes per entry to be realistic. In a DynamoDB table with a TTL attribute, an item with a single binary partition key plus the TTL number runs roughly 40–50 bytes of billable size plus internal index overhead; call it ~50 bytes. Let us compute both.

At 80 bytes per entry (Redis-like): `30.24e9 × 80 bytes = 2.419e12 bytes ≈ 2.42 TB` of RAM. That is not a typo — keeping 30 billion keys for 7 days in an in-memory store needs on the order of **2.4 terabytes of RAM**, which across a Redis cluster is dozens of large nodes and a meaningful monthly bill. At 50 bytes (DynamoDB-like on-disk): `30.24e9 × 50 = 1.512e12 bytes ≈ 1.51 TB` of stored data, plus the write cost — 50,000 writes per second sustained is 50,000 write-capacity-units, which on-demand is a substantial recurring charge.

The lesson is twofold. First, the dedup store at high throughput is *not* a footnote — it is a 1-to-2.5 TB system in its own right, and you must budget for it. Second, this is exactly why the window length is an engineering decision and not a default: cutting the window from 7 days to 1 day cuts the key count by 7x, to about 4.3 billion keys and ~345 GB at 80 bytes — a far more comfortable footprint. So you cut the window to the shortest value that still covers your real redelivery sources. If your only duplicates come from retries within minutes, a 1-hour window holds `50,000 × 3,600 = 180 million` keys, about **14.4 GB** at 80 bytes — trivial. The same throughput spans a four-order-of-magnitude storage difference depending purely on whether you must defend against multi-day replays or only against minute-scale retries. *Measure your real redelivery gap; do not pay for a 7-day window if your duplicates all arrive within the hour.*

### Compacting the cost with bloom filters

When the key count is too large to store exactly, a bloom filter trades a small false-positive rate for a massive space saving. A bloom filter sized for 30 billion entries at a 0.1% false-positive rate needs roughly `30e9 × 1.44 × log2(1/0.001) / 8 ≈ 30e9 × 1.44 × 9.97 / 8 ≈ 53.8 GB` — versus the multi-terabyte exact store. The catch is the false positive: a 0.1% rate means one in a thousand *new* messages is wrongly judged a duplicate and skipped, which for a side effect that must never be skipped is a one-in-a-thousand silent message loss. So bloom filters are appropriate for *best-effort* dedup (suppress most repeats cheaply, accept rare over-suppression) but not for operations where dropping a legitimate message is unacceptable. The honest layering is a bloom filter as a cheap pre-filter — "definitely new" messages skip the expensive exact check, "maybe seen" messages fall through to the exact store — which keeps the exact store's read load down while preserving exact correctness.

## 7. Where to deduplicate: producer, broker, consumer

Deduplication can run at three layers — at the producer before the message is even sent, at the broker as messages pass through, or at the consumer where the work is done. The figure below stacks them, and the punchline is in the colors: producer and broker dedup are partial and fragile, while the consumer is the layer that owns the side effect and is therefore the honest place to make it safe.

![A stack diagram showing three deduplication layers — producer best-effort, broker windowed, and consumer owning the side effect, backed by a durable dedup store](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-4.webp)

### Producer-side dedup

Producer dedup means the producer refuses to send a message it has already sent — the [idempotent producer in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions), or your own logic that checks "did I already publish this event?" before publishing. The Kafka idempotent producer (`enable.idempotence=true`) is genuinely valuable: it stops *producer retries* from creating duplicate records in the log by assigning each producer a producer id and a per-partition sequence number, so the broker rejects a resend of a sequence it has already seen. This eliminates one important *source* of duplicates — the producer-retry source — at the log level. But it is partial in two ways. First, it only dedups within a single producer session on a single partition; a producer restart gets a new producer id and loses the sequence state, so a message resent across a restart is not caught. Second, and more fundamentally, it does nothing about *downstream* duplicates — rebalances, consumer redelivery, application retries all still produce repeats that the producer never sees. Producer dedup narrows the funnel; it does not close it.

### Broker-side dedup

Some brokers offer built-in deduplication over a window. SQS FIFO queues dedup on a `MessageDeduplicationId` (or a content hash) within a fixed **5-minute** window. Pulsar offers broker-side dedup keyed on a sequence id. These are useful for catching the clustered-in-time duplicates — a producer retry storm within the 5-minute window is suppressed by the broker, and your consumer never sees it. But broker dedup is bounded by exactly the window-sizing constraint of the previous section, and the broker's window is *fixed and short* (5 minutes for SQS FIFO), which covers retry storms but not the longer redelivery gaps from rebalances minutes later or replays hours later. Broker dedup is also scoped to the broker's own view: it cannot know that two messages with different ids are the same business operation, because it does not understand your domain. It dedups identical sends, not equivalent operations.

### Consumer-side dedup is the honest place

Here is the recommendation, stated plainly: **the consumer is almost always the right place to deduplicate, because the consumer is the only layer that owns the side effect.** The producer can stop itself from sending twice but cannot stop a rebalance from redelivering. The broker can suppress identical sends within a short window but cannot understand your business notion of "same operation" and cannot extend its window to cover your real redelivery gap. Only the consumer sits at the exact point where the side effect happens — the database write, the card charge, the email — and only the consumer can make that side effect idempotent with respect to a domain-meaningful key over a window you control. Every other layer is a partial optimization that *reduces* the duplicate rate the consumer must handle, which is welcome, but none of them *eliminates* the consumer's responsibility. Design the consumer to be idempotent first; treat producer and broker dedup as cheap pre-filters that lighten its load, never as a reason to skip it.

The deeper principle is about *ownership*. The side effect is owned by the code that performs it. Safety properties are most robust when enforced at the point of ownership, because that point has the full context — the transaction, the domain key, the actual write — needed to make the property hold atomically. Pushing dedup to a layer that does not own the side effect means that layer is making a promise it cannot fully keep, and the gap between the promise and the reality is where the double-charge lives. Own the side effect, own the dedup.

It helps to quantify how the layers stack in practice, because "the consumer is responsible" does not mean the upstream layers are useless. Picture a stream where the raw duplicate rate, before any dedup, is one in a thousand deliveries. Turn on the Kafka idempotent producer and you eliminate the producer-retry duplicates — say that is 40% of them — dropping the rate the consumer sees to six in ten thousand. Add SQS-FIFO-style broker dedup over a five-minute window and you suppress the clustered retry-storm duplicates — say another 40% — dropping it to roughly three in ten thousand. The consumer's durable dedup store then has to catch only that residual three-in-ten-thousand, which is the rare, spread-out, cross-restart kind that nothing upstream can see. The upstream layers did real work: they cut the consumer's dedup-store read-and-write load by more than half, which at fifty thousand messages a second is a meaningful reduction in database pressure. But notice the consumer still had to be there, because the residual duplicates are exactly the ones that survive every upstream filter. The layers are multiplicative pre-filters in front of one mandatory backstop, not substitutes for it.

A practical corollary: instrument the duplicate rate *at the consumer's dedup store* — count how often a lookup is a hit. That hit rate is the single best health signal for this whole machine. A hit rate that suddenly climbs from three-in-ten-thousand to three-in-a-hundred means something upstream is redelivering far more than usual — a rebalance loop, a stuck consumer exceeding its poll interval, a producer retrying into a slow broker. The dedup store, sitting at the point of ownership, is the only place that sees the *true* end-to-end duplicate rate after every upstream filter, which makes it both your safety mechanism and your best observability surface for delivery health.

## 8. The hard case: idempotency across external side effects

Everything so far has a clean solution when the side effect lands in *your* database, because your database gives you transactions, unique constraints, and conditional writes — atomic primitives you can lean on. The genuinely hard case is when the side effect leaves your system entirely: you charge a card through Stripe, send an email through SES, post a webhook to a partner, enqueue an SMS through Twilio. Now the effect happens in a system you do not control, with no shared transaction, and your local dedup table cannot make the *remote* effect idempotent. The card gets charged in Stripe's ledger, not yours; your rolling back your transaction does not un-charge it.

This is where two specific techniques earn their keep, and the figure below shows them working together: a provider idempotency key for the external system, and a local dedup record that ties the two together.

![A graph showing an idempotent receiver that checks a local dedup store and calls a payment API with an idempotency key so the external charge lands once despite consumer retries](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-9.webp)

### Provider idempotency keys

The first and most important technique is to push the dedup responsibility *to the provider* using its idempotency-key mechanism. Every serious provider of an external side effect offers one. Stripe accepts an `Idempotency-Key` header on every mutating request; if you send the same key twice, Stripe performs the operation once and returns the *original* response to the duplicate. SES, PayPal, Adyen, and most modern payment and messaging APIs have an equivalent. The contract is exactly the idempotency we have been building toward: *you* generate a stable key for the logical operation, *they* guarantee the operation happens once per key. The key must be — again — stable across your retries: you generate it once when the logical charge is born (tie it to your order id or payment id), persist it, and pass the *same* key on every retry of that charge. A consumer that crashes after charging but before recording the result will retry, send the same idempotency key, and Stripe will return the original successful charge instead of charging again. The double-charge is impossible *at the provider*, which is the only place it can be made impossible, because the provider owns the ledger.

```python
# Stable idempotency key tied to the order, persisted, reused on every retry
def charge_for_order(order):
    # generated ONCE when the charge is born, stored with the order
    idem_key = order.payment_idempotency_key  # e.g. "charge-order-8842"
    resp = stripe.PaymentIntent.create(
        amount=order.total_cents,
        currency="usd",
        customer=order.stripe_customer_id,
        idempotency_key=idem_key,   # same key on retry -> same charge, once
    )
    return resp  # on a duplicate call, Stripe returns the ORIGINAL intent
```

The discipline that makes this work is generating the key *before* the first attempt and persisting it, so that a crash between generation and the call, or between the call and recording the result, still retries with the identical key. If you generate the key inside the retry, you generate a new one each time and the provider sees distinct operations. The provider idempotency key is the single most important tool for external side effects, and the rule is one sentence: *one stable key per logical external operation, generated and persisted before the first attempt, reused unchanged on every retry.*

### The dual-write problem and why ordering matters

Even with a provider idempotency key, there is a residual hazard: your consumer must do *two* writes that are not in the same transaction — the external charge (in the provider) and the local record that the charge happened (in your database). These are a classic dual write, and they cannot be made atomic because they live in different systems. The question is the *ordering*, and there is a correct answer. You must call the external side effect *first* and record locally *second*, never the reverse. Here is why: if you record locally first and then crash before the external call, your local state says "charged" but no charge ever happened — a silent under-charge, money you will never collect, and worse, a dedup record that will *suppress* the retry that would have fixed it. If instead you charge first and crash before recording locally, the retry re-charges with the same idempotency key, the provider returns the original charge (no double charge), and you finally record it. Charging first is safe because the provider's idempotency key makes the repeated charge a no-op; recording first is unsafe because nothing makes the *missing* charge happen. The rule: *do the idempotent external effect first, record the outcome second; lean on the provider key to make the repeated effect harmless.*

This ordering depends critically on the external effect being idempotent via the provider key. If the provider has *no* idempotency mechanism — some legacy or partner APIs do not — you are in the worst spot in distributed systems, and there is no fully safe answer. You must choose between at-most-once for that effect (record an intent, call once, never retry, accept occasional loss) and at-least-once (retry, accept occasional duplicates and reconcile them out-of-band). For irreversible non-idempotent external effects with no provider key, the honest engineering move is to build a reconciliation process — periodically compare your records against the provider's and detect and correct the discrepancies — because prevention is impossible and detection is the best you can do. When you can, choose providers that offer idempotency keys; it is a first-class selection criterion for anything that touches money.

#### Worked example: a payment charged twice, made safe

Let us walk the opening war story end to end with numbers, because it crystallizes every concept. An order for **\$50** flows through an at-least-once pipeline. The naive consumer does: `process` → `charge card` → `commit offset / ack`. Here is the failure trace, second by second.

At t=0, the consumer receives the message for order 8842. At t=0.1s it calls the payment API and the card is charged \$50 successfully. At t=0.3s, before the offset commit at t=0.4s could run, the pod receives a SIGTERM from a routine deploy and dies. The offset was never committed. At t=2s a new pod joins the consumer group, a rebalance assigns it the partition, and it re-reads from the last committed offset — which is *before* order 8842. At t=2.1s it processes order 8842 again and charges the card a *second* \$50. Net effect: the customer is charged **\$100** for a \$50 order. Multiply by the fraction of in-flight messages caught by any given deploy across a busy fleet and you get the two thousand double-charges from the story. The root cause is not a bug in any component; it is the absence of idempotency at the one place — the card charge — that has a side effect the pipeline cannot undo.

Now make it safe with two changes. First, when the order is created, generate and persist a stable idempotency key: `idem_key = "charge-order-8842"`. Second, restructure the consumer to charge first with that key, then record. Replay the exact same failure. At t=0.1s the consumer charges Stripe with `Idempotency-Key: charge-order-8842`; the card is charged \$50; Stripe records the key. At t=0.3s the pod dies before recording locally. At t=2.1s the new pod reprocesses order 8842 and calls Stripe *again* with the identical `Idempotency-Key: charge-order-8842`. Stripe recognizes the key, performs *no* second charge, and returns the original \$50 PaymentIntent. The consumer records the result and acks. Net effect: the customer is charged **\$50, exactly once**, despite the message being delivered and processed twice. The total system change was one persisted string and a reordering of two lines, and it converted a \$100 double-charge into a correct \$50 charge. That is the entire return on investment of idempotency, and it is enormous relative to its cost.

## 9. The inbox pattern and effectively-once

We have one loose end: the dual-write race *inside* your own system, between recording the dedup key and doing the business write, when both land in your database. The unique-constraint strategy from section 5 already solves the case where the business write is a single transactional write — put the dedup key insert in the same transaction and the database makes it atomic. The inbox pattern generalizes this into a clean, named architecture that is worth knowing by name because it is the canonical solution and it pairs symmetrically with the outbox pattern on the publish side.

The inbox pattern works like this, and the figure below traces it as a grid. When a message arrives, you insert it (or just its idempotency key) into an **inbox table** that has a unique constraint on the key, *in the same local transaction* as the business write. The first delivery inserts the inbox row and does the business write atomically and commits. A redelivery's insert hits the unique constraint, the entire transaction — including the business write — rolls back, and you treat it as a successful duplicate. Because the inbox insert and the business write share one ACID transaction, there is no window where one happens without the other, and the dedup is exact. You then mark the inbox row done (or simply let its presence serve as the done-marker) and acknowledge the message.

![A grid diagram of the inbox pattern showing a message arriving, inserting into an inbox table where a duplicate id is rejected, marking done within the same transaction, processing with a local write, and committing atomically](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-8.webp)

```python
def handle(msg):
    key = msg.headers["idempotency-key"]
    try:
        with db.transaction():
            # inbox insert + business write share ONE atomic transaction
            db.execute(
                "INSERT INTO inbox (message_id, received_at) VALUES (%s, now())",
                (key,),
            )
            apply_business_effect(msg)   # e.g. upsert order, decrement stock
        ack(msg)                          # commit succeeded -> safe to ack
    except UniqueViolation:
        ack(msg)                          # duplicate: already processed, ack and move on
```

Notice the two `ack` calls do different jobs. After a successful commit, the ack tells the broker the work is durably done. After a `UniqueViolation`, the ack tells the broker this duplicate is handled — the work was done by the *first* delivery, so acking the duplicate is correct and prevents an infinite redelivery loop. Both paths end in an ack; the difference is whether *this* delivery did the work or a prior one did. That symmetry — every delivery, original or duplicate, ends in exactly one durable effect and one ack — is the signature of a correctly idempotent consumer.

### The inbox table's relationship to the dedup window

The inbox table is a dedup store, so everything from the window section applies: you must keep each inbox row at least as long as the longest redelivery gap, and you reclaim space by deleting rows older than the window (a periodic `DELETE FROM inbox WHERE received_at < now() - interval '7 days'`, or a partition-drop on a time-partitioned inbox table for cheaper bulk reclaim). The inbox table is also where the storage math of the worked example lands in practice: at 50k msg/s with a 7-day window, that inbox table holds tens of billions of rows, which is why time-partitioning and partition-drop reclaim — rather than row-by-row deletes — is the operationally sane way to age it out.

### Effectively-once: the only exactly-once you can ship

We can now state the equation this whole post has been building toward, and it is the same one the delivery-semantics post promised this post would prove out:

**effectively-once = at-least-once delivery + idempotent processing.**

There is no exactly-once delivery — the network forbids it. What you can build, and what every system that advertises "exactly-once" actually ships, is *effectively-once processing*: the message may be delivered any number of times (at-least-once handles the loss problem by retrying), and the processing is idempotent (which handles the duplicate problem by making repeats invisible). The two halves are complementary and both necessary. At-least-once without idempotency gives you no loss but double effects — the broken naive consumer. Idempotency without at-least-once gives you no double effects but possible loss — you would not retry, so a genuinely-lost message stays lost. Together they give you the property everyone actually wants: every message's effect lands exactly once, achieved not by delivering once but by *applying* once. The figure at the very top of this post is this equation in pictures: the at-least-once redelivery is the same in both columns; idempotency is the only difference, and it is the difference between a double charge and a correct one.

The taxonomy figure below organizes every approach we covered under this one goal. Idempotency is reached either *naturally* — the operation is already shaped right (SET, upsert, compare-and-set) — or by *keying* — you add a stable dedup key and a store to recognize repeats (dedup table, unique constraint, inbox pattern). Natural idempotency is cheaper and should be your first move; keyed idempotency is more general and is what you reach for when nature does not cooperate, which is most of the time for real business operations.

![A tree diagram organizing idempotency approaches into natural operation-level methods like SET and compare-and-set and keyed methods like dedup tables, unique constraints, and the inbox pattern](/imgs/blogs/idempotency-and-deduplication-making-at-least-once-safe-5.webp)

## Case studies and war stories

### The double-charge deploy (payments)

The opening story is the canonical case and worth distilling into its lesson. A payments consumer charged cards before committing offsets, and a routine deploy's SIGTERM landed in the window between charge and commit on a few thousand in-flight messages, double-charging every one of them on redelivery. Nothing was "broken" — every component honored its contract. The fix had two parts and both mattered. First, a persisted Stripe idempotency key tied to the order, so the *repeated* charge became a no-op at the provider. Second, reordering the consumer to charge-then-record, so a crash always retried into the idempotent path rather than recording a phantom success. The lesson: **at-least-once plus an external side effect is a double-charge generator until you add a provider idempotency key, and the key must be persisted before the first attempt so the retry reuses it.** Teams relearn this every time they add a new payment integration without checking the idempotency-key support; make "does the provider support idempotency keys" a gate in your vendor selection for anything touching money.

### The expired dedup window (notifications)

A notifications team built a clean consumer-side dedup table in Redis with a TTL — but they set the TTL to **one hour**, reasoning that retries always happen within minutes. That held for a year. Then an incident took their consumers down for six hours, they replayed the backlog from the broker, and every message older than one hour had already had its dedup key expire — so the replay re-sent **hundreds of thousands of duplicate push notifications and emails** to users, because the dedup table had forgotten the keys. The window was sized for the *common* redelivery source (minute-scale retries) but not the *worst* one (a multi-hour outage followed by a replay). The lesson: **size the dedup window to the maximum age at which a redelivery is possible, not the typical one — and a backlog replay after an outage is a redelivery source that can be hours or days old.** They moved to a 7-day window on a durable store and added a circuit breaker that pauses notification side effects during a mass replay, so a backfill can be reprocessed without spamming users.

### The non-idempotent metric (analytics)

An analytics pipeline incremented per-user counters from a Kafka stream — `INCR user:123:events` on each message. It looked harmless because counters "self-heal," except they do not: every rebalance reprocessed a few hundred messages and inflated the counters by the reprocessed count, and over months the counts drifted **8–12% high** with no single visible bug, just a steady upward bias that corrupted every downstream report and every billing calculation derived from those counts. The fix was to make the write idempotent: instead of incrementing on each event, the producer stamped each event with a stable event id, and the consumer recorded `(user_id, event_id)` in a dedup set and only incremented on first sight — or, better, switched to recomputing the absolute count from a deduplicated event store. The lesson: **a non-idempotent increment under at-least-once does not produce a loud error; it produces a quiet, compounding bias that corrupts derived data and is nearly impossible to attribute after the fact.** The most dangerous non-idempotency is the kind that does not crash.

### The dual-write that lost charges (e-commerce)

An e-commerce team, having learned about idempotency keys, added a dedup table and recorded the key *before* calling the payment provider, reasoning "record that I'm handling this, then do the work." A crash between the dedup-record and the charge left a dedup key with no charge behind it — and because the dedup key was now present, the redelivery saw it, concluded "already processed," skipped the charge, and the order shipped *unpaid*. They had inverted the safe ordering. The fix was to charge first (with the idempotency key making a repeat safe) and record second, so a crash always retried into a re-charge that the provider deduplicated, rather than into a skip. The lesson: **with a non-transactional external effect, record the dedup key *after* the idempotent effect succeeds, never before — recording first turns a crash into a silently skipped side effect.** The ordering of the dedup write relative to the side effect is as load-bearing as the ack timing in the delivery-semantics post; get it backwards and idempotency becomes message loss.

## When to reach for this (and when not to)

Reach for idempotency and deduplication essentially **always**, because at-least-once is essentially always your delivery semantic, and the only question is *which* mechanism, not *whether*. The decision tree is short. If you can reshape the operation to be naturally idempotent — an absolute SET, an upsert keyed on a business id, a compare-and-set — do that first, because it costs nothing at runtime and there is no separate store to operate. If the business write is transactional but not naturally idempotent, add a unique constraint on the dedup key inside the business transaction, or use the inbox pattern — atomic, exact, and the gold standard for database-bound work. If the side effect leaves your system, use the provider's idempotency key as the primary defense, charge-then-record, and add reconciliation for any provider that lacks a key. Layer a cheap in-memory idempotent-receiver in front of the durable store to absorb retry storms without database load, and treat producer and broker dedup as welcome pre-filters that lighten the consumer's job.

When can you skip it? Only when the operation is *already* idempotent by its nature and you have proven it — a pure SET to a value derived solely from the message, where no concurrent writer can race you, genuinely needs no dedup store. And when the cost of a duplicate is *provably zero* — some read-only or cache-warming consumers can reprocess freely. But be suspicious of "duplicates are harmless here" claims; the analytics war story is what happens when a team believes that about an increment. The honest default is: **assume duplicates will happen, prove your handling is idempotent, and only then relax the mechanism if the operation truly does not need it.** Skipping idempotency to save a write is a false economy that you pay back with interest the first time a deploy lands in the wrong millisecond.

A note on where *not* to over-engineer: do not try to achieve exactly-once *delivery* — it does not exist, and chasing it leads to brittle coordination protocols that are slower and less reliable than at-least-once plus idempotency. Do not build a global "have we seen this message" service that every consumer must consult synchronously — it becomes a coordination bottleneck and a single point of failure, and it violates the locality principle that makes idempotency robust. Keep dedup local to the consumer that owns the side effect, keep the window sized to your real redelivery gap and no longer, and let the provider own dedup for effects that leave your system. The discipline is to make safety *local and cheap*, not *global and coordinated*.

## Key takeaways

- **At-least-once makes duplicates a steady-state rate, not a rare event.** A lost acknowledgement is indistinguishable from a lost message, so the broker retries whenever unsure, and it is unsure constantly. Build a mechanism that runs on every message, because you cannot know which one is the repeat.
- **Idempotency means applying twice equals applying once** — precisely, the observable state after N applications is identical to the state after one. It is a *local* property you establish at the point of the side effect, not a global agreement you negotiate with the network.
- **Prefer natural idempotency first.** Absolute SETs, upserts keyed on a business id, and compare-and-set writes are idempotent for free. Reformulate increments into absolute sets by carrying the resulting value in the message rather than the delta.
- **A stable dedup key is the whole game.** Use a business key (order id, payment id) or a producer-stamped message id generated once at creation and reused on every redelivery. Never use a receive-time UUID, a timestamp, or a broker offset — they are not stable across redeliveries.
- **A unique constraint inside the business transaction is the gold standard** for database-bound work: the database enforces exactly-once-per-key atomically, with no race to reason about. The inbox pattern is this generalized into a named architecture.
- **Size the dedup window to the longest possible redelivery gap, plus margin** — broker retention plus retry backoff plus dead-letter replay time. A backlog replay after an outage is a multi-hour redelivery source that short windows miss.
- **At 50k msg/s a 7-day window holds ~30 billion keys and 1.5–2.5 TB.** The dedup store is a real system to budget for; cut the window to the shortest value that covers your actual redelivery sources, since 1-hour versus 7-day is a 168x storage difference.
- **The consumer is the honest place to deduplicate** because it owns the side effect. Producer and broker dedup are partial pre-filters that reduce the rate but never eliminate the consumer's responsibility.
- **For external side effects, push dedup to the provider's idempotency key,** generate and persist it before the first attempt, reuse it unchanged on retries, and do the external effect *before* recording locally so a crash retries into an idempotent no-op rather than a skipped charge.
- **Effectively-once = at-least-once + idempotency.** There is no exactly-once delivery; there is only exactly-once *effect*, achieved by delivering many and applying once. Every system that advertises exactly-once ships this.

## Further reading

- [Delivery semantics: at-most-once, at-least-once, and the exactly-once myth](/blog/software-development/message-queue/delivery-semantics-at-most-at-least-exactly-once) — the layer below this post, where ack timing determines whether you lose or duplicate.
- [The transactional outbox pattern](/blog/software-development/message-queue/transactional-outbox-pattern-reliable-publishing) — the publish-side mirror of the inbox pattern, making your *outgoing* events reliable and idempotent.
- [Exactly-once in Kafka](/blog/software-development/message-queue/exactly-once-in-kafka-idempotent-producer-transactions) — how the idempotent producer and transactions wire the producer and consumer sides together.
- [Change data capture and the outbox pattern](/blog/software-development/database/change-data-capture-and-the-outbox-pattern) — capturing state changes reliably as events, a common source of the messages you must dedup.
- [Consistency models: from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) — the consistency context that decides whether your dedup store reads are fresh enough to be trusted.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — retention, offsets, and replay, which set the dedup window you must cover.
- Stripe API documentation, "Idempotent Requests" — the canonical provider-side idempotency-key contract.
- Gregor Hohpe and Bobby Woolf, *Enterprise Integration Patterns* — the Idempotent Receiver and related messaging patterns.
