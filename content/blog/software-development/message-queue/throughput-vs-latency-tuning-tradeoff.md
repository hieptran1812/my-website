---
title: "Throughput vs Latency: The Fundamental Tuning Tradeoff in Message Systems"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "Master the one tradeoff that governs every tuning decision in a message system: why batching buys throughput at the cost of latency, what Little's Law and queueing theory tell you about saturation, why the average latency lies, and how to pick a target and tune toward it instead of chasing both."
tags:
  [
    "message-queue",
    "throughput",
    "latency",
    "performance-tuning",
    "queueing-theory",
    "kafka",
    "rabbitmq",
    "distributed-systems",
    "event-driven",
    "tail-latency",
  ]
category: "software-development"
subcategory: "Message Queue"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/throughput-vs-latency-tuning-tradeoff-1.webp"
---

There is a single tradeoff sitting underneath almost every performance knob in a message system, and once you see it you cannot unsee it. Every time someone asks "how do I make Kafka faster?" or "why is RabbitMQ slow?" or "should I increase the batch size?", they are, whether they know it or not, asking a question about this one tradeoff. Faster how? Faster as in more messages per second, or faster as in each individual message arrives sooner? Those two notions of speed are not the same thing. They are frequently in direct opposition. You can almost always buy more of one by spending some of the other, and the great majority of tuning mistakes I have watched engineers make — including mistakes I have made myself, in production, at three in the morning — come from not having decided which one they actually wanted before they started turning dials.

Throughput is rate: messages, bytes, or requests completed per unit of time. Latency is delay: how long one individual message waits from the moment it is produced until the moment it is fully processed. A system can have enormous throughput and terrible latency at the same time — a nightly batch job that moves a billion records but takes six hours is exactly that. A system can have beautiful latency and pathetic throughput — a hand-tuned low-frequency trading path that gets one message through in eight microseconds but would fall over at ten thousand messages a second. Most real systems live somewhere in the messy middle, and the entire discipline of tuning a message system is the discipline of deciding where in that middle you want to sit, and then moving there deliberately instead of by accident.

The figure below shows the same producer, configured two ways, landing in two completely different places on the map. On the left it is tuned for latency: tiny batches, no artificial waiting, every message flushed the instant it arrives. On the right it is tuned for throughput: large batches, a deliberate pause to let batches fill, far fewer but far fatter requests crossing the network. Same code, same broker, same hardware. The only difference is which axis the operator decided mattered. By the end of this post you will be able to look at any message-system config — a Kafka producer, a RabbitMQ channel, a consumer poll loop, a broker's replication settings — and predict which way each setting bends this curve, and you will have a methodology for picking your target and tuning toward it rather than thrashing between the two.

![A before and after comparison showing a latency-optimized producer with tiny low-linger batches reaching low per-message delay versus a throughput-optimized producer with large high-linger batches reaching far higher message rate](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-1.webp)

This is the master tuning post in the series, and it is deliberately placed before the two posts that go deep on the specific knobs: [producer-side tuning](/blog/software-development/message-queue/producer-optimization-batching-compression-acks) and [consumer-side tuning](/blog/software-development/message-queue/consumer-optimization-and-scaling). Those posts tell you which dial to turn. This one tells you which direction is "better" and why, so that when you read them you already know what you are trading. If you have not yet read the [anatomy of a message system](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) or the opener on [what a message queue is for](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling), they give you the vocabulary — producer, broker, partition, offset, consumer group — that this post assumes. Everything here is about how fast that machinery runs, and at what cost.

## 1. The tradeoff, stated plainly

Let us be precise about the two quantities, because sloppiness here is the root of most confusion.

**Throughput** is a rate. It is measured in things-per-second: records per second, bytes per second, transactions per second. When you say "this cluster does 800,000 messages a second," you are stating throughput. Throughput is a property of the *system over an interval*. It does not tell you anything about how any individual message fared. A system that processes a million messages in a one-second burst and then sits idle has the same one-second throughput as a system that processes a million messages perfectly smoothly over that second, even though the two would feel utterly different to a user waiting on a specific message.

**Latency** is a duration. It is measured in time-per-thing: milliseconds per message, microseconds per request. When you say "p99 produce latency is 12 milliseconds," you are stating latency. Latency is a property of an *individual message's journey*. Crucially, latency is not one number — it is a distribution. Some messages are fast, some are slow, and as we will see, the slow ones are usually the ones that matter.

Here is the relationship that makes them a tradeoff rather than two independent dials. Most of the techniques that raise throughput work by **amortizing fixed per-operation overhead across more work**. A network round-trip costs roughly the same whether you send one message or a thousand in it. An `fsync` to disk costs roughly the same whether it flushes one record or a megabyte of records. A syscall to read from a socket costs the same whether it returns one byte or sixty-four kilobytes. So if you can group operations together — batch them — you pay the fixed cost once instead of N times, and your throughput goes up, sometimes by one or two orders of magnitude.

But grouping requires *waiting*. To put a thousand messages in one network request, you have to hold the first message back until the thousandth one shows up, or until some timer tells you to give up waiting and send what you have. That holding-back is added latency, paid by every message except the last one to join the batch. The first message in a batch of a thousand, which could have been sent immediately, instead sits in a buffer waiting for nine hundred and ninety-nine friends to arrive. The throughput you gained by batching was *bought with the latency you added by waiting*. This is the heart of the tradeoff, and it is mechanical, not a matter of opinion. You cannot batch without waiting, and you cannot wait without adding latency.

The same logic recurs at every layer of the stack, which is why this is the master tradeoff. Compression amortizes I/O and network cost across a block of data — but you have to spend CPU time, and CPU time is latency. Replication with `acks=all` amortizes durability guarantees by confirming a whole batch is on multiple disks — but waiting for the slowest replica is latency. Larger consumer fetches amortize the poll syscall across more records — but `fetch.min.bytes` makes the broker hold the response until enough data accumulates, which is latency. Every single one of these is the same shape: spend latency, gain throughput, or spend throughput, gain latency. Once you internalize the shape, you stop being surprised by it.

### Why you cannot just have both

The naive hope is that with enough hardware you escape the tradeoff entirely — throw money at it, buy faster disks and a fatter network, and get both high throughput and low latency. Hardware does help, and it raises the whole curve, but it does not eliminate the tradeoff. The reason is that the tradeoff is not fundamentally about how fast your hardware is; it is about *when you choose to do work*. Batching is a decision to defer work so you can do it in bulk. Deferral is latency, by definition, no matter how fast the eventual bulk operation runs. You can make the bulk operation faster with better hardware, which shrinks one term, but the deferral term — the wait — is a choice you made on purpose to get the amortization, and faster hardware does not give you that amortization for free. If you want the per-message latency of not-batching, you have to actually not batch, and then you pay the per-message overhead you were trying to amortize away.

So the honest framing is this: pick a target. Are you building something latency-sensitive, where a user or an upstream system is blocked waiting on each message? Are you building something throughput-sensitive, where the goal is to move a mountain of data and nobody is staring at any single record? Or are you balanced, needing decent latency under decent load without optimizing hard for either extreme? You answer that question first, and then every knob has a correct direction. Answer it wrong, or fail to answer it, and you will tune in circles.

## 2. Batching: the canonical throughput-vs-latency knob

Batching is the cleanest, most important instance of the tradeoff, so we will spend real time on it. Understand batching deeply and the rest of the knobs become variations on a theme you already know.

A producer does not have to send each message to the broker the instant your code calls `send()`. In fact, in any high-performance client, it almost never does. Instead, the client accumulates messages in an in-memory buffer, grouped by destination partition, and sends them as a **batch** — a single request carrying many records. Two parameters govern when a batch gets flushed to the network: the **batch size** (a byte cap — flush when the batch reaches this many bytes) and the **linger** (a time cap — flush when the batch has been open this long, even if it has not reached the size cap). The batch flushes on whichever trigger fires first.

The matrix below summarizes how each major knob pushes on the two axes. I want you to look at it before we dig in, because the pattern it shows — that the throughput column is mostly wins and the latency column is mostly costs — is exactly the tradeoff made visual. There is no row where you win on both.

![A matrix mapping each tuning knob such as bigger batch size higher linger more parallelism compression and larger buffers against its effect on throughput and on latency showing throughput gains paired with latency costs](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-2.webp)

### Linger: buying throughput with a timer

Linger is the purest expression of the tradeoff, so let us be exact about what it does. With linger set to zero — Kafka's default `linger.ms=0` — the producer sends a batch as soon as it has anything to send and a sender thread is free. In practice this still batches a little under load, because while one request is in flight, messages pile up and the next request scoops them all, but at low to moderate rates each message goes out nearly alone. Latency is minimal; throughput is capped by how many tiny requests the broker can handle per second, which is far fewer than the raw byte rate it could sustain.

Set `linger.ms=10` and the producer will hold a batch open for up to ten milliseconds, hoping more messages arrive to fill it. At a high enough message rate the batch fills on bytes before the ten milliseconds elapse, so you pay little or no extra latency and reap the full batching benefit. At a low rate, you pay up to the full ten milliseconds of waiting per message and gain little, because there were not enough messages to make a meaningful batch anyway. This is the key non-obvious insight: **linger's latency cost is highest exactly when its throughput benefit is lowest**, and vice versa. At high load, linger is almost free; at low load, it is almost pure cost. We will make this concrete with numbers shortly.

The timeline figure shows a single batch's life. A message arrives and opens the batch. More messages join over the next few milliseconds. The batch hits its byte cap before the linger timer expires, so it flushes early — the size trigger won the race. The linger timer, which would have fired later, never gets the chance. This is the common case under healthy load, and it is why a well-tuned system with a non-zero linger often shows almost no latency penalty: the batches are filling on size, not on time.

![A timeline showing a batch opening when the first message arrives then more messages joining until the batch reaches its byte cap and flushes early before the linger timer would have fired](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-3.webp)

### The batching math

Let us make the tradeoff quantitative, because intuition without numbers leads people to set `linger.ms` to absurd values.

#### Worked example: linger 0ms vs 10ms at 5,000 messages per second

Suppose your producer is sending 5,000 messages per second to a single partition, and each message is 200 bytes. The producer's batch size cap is 256 KB (262,144 bytes), the Kafka default after a bump.

**With `linger.ms=0`:** The producer sends as fast as the sender thread cycles. Suppose the round-trip to the broker — send the request, get the ack back — is 2 ms. While that request is in flight, messages accumulate. At 5,000 msg/s, in 2 ms about 10 messages arrive (5,000 × 0.002). So each request carries roughly 10 messages × 200 bytes = 2,000 bytes, and the producer sends about 500 requests per second (5,000 messages ÷ 10 per request). The added latency from batching is essentially zero — a message waits only as long as the in-flight request ahead of it, which is the unavoidable round-trip, not an artificial linger. The cost is 500 requests/second hammering the broker, each tiny.

**With `linger.ms=10`:** Now the producer holds each batch open for up to 10 ms. In 10 ms, at 5,000 msg/s, about 50 messages arrive (5,000 × 0.010). Each is 200 bytes, so a batch is roughly 50 × 200 = 10,000 bytes — well under the 256 KB cap, so the batch flushes on the linger timer, not on size. The producer now sends about 100 requests per second (5,000 ÷ 50 per request) instead of 500. That is a 5x reduction in request count, which is a real, direct reduction in broker CPU, syscalls, and network framing overhead. The cost: each message now waits on average about 5 ms longer (half of the 10 ms linger window, since messages arrive uniformly across the window), and up to 10 ms in the worst case. So you traded roughly 5 ms of average added latency for an 80 percent cut in request rate.

Whether that trade is good depends entirely on your target. If you are feeding an analytics pipeline where nobody notices 5 ms, it is a free win — take it. If you are in a request-reply path where a user is blocked on each message and your latency budget is 15 ms end to end, spending 5 of those 15 milliseconds on linger alone is reckless. Same knob, opposite verdict, and the only thing that changed was the target.

#### Worked example: how much throughput does a larger batch actually buy

Now flip it and look at the throughput side. Say each network request to the broker has a fixed overhead — request framing, the broker's per-request handling, the round-trip — that costs the equivalent of 0.5 ms of broker time regardless of payload, plus the broker can write payload bytes at, say, 200 MB/s once it is handling the request.

With 2,000-byte batches (the `linger.ms=0` case above), the broker spends 0.5 ms of overhead to write 2,000 bytes. The overhead utterly dominates: 0.5 ms is the equivalent of 100,000 bytes of write time at 200 MB/s, so the broker is spending 98 percent of its effort on overhead and 2 percent on actual data. With 100,000-byte batches, that same 0.5 ms of overhead now amortizes across 100 KB of useful payload, so overhead and data are roughly balanced, and the broker's effective useful throughput per unit of its own work jumps by something like 50x. This is why batching matters so much: at small batch sizes you are paying fixed costs over and over, and the broker spends almost all its energy on bookkeeping rather than moving your data. The first few kilobytes of batch size buy you enormous throughput; past a point, the curve flattens because the fixed overhead is already well amortized and you are mostly just adding latency for diminishing return. This is why "make the batch bigger" stops helping after a while and starts only hurting latency — a fact we will return to in the tuning methodology.

### The two configs, in code

The cleanest way to see the tradeoff is to lay the two extreme configs side by side. Here is a Kafka producer tuned for latency, and the same producer tuned for throughput. Notice that nearly every setting that helps one hurts the other — the configs are almost mirror images, which is the tradeoff made literal in code.

```java
// Latency-optimized producer: every message goes out immediately.
Properties latency = new Properties();
latency.put("bootstrap.servers", "broker1:9092");
latency.put("linger.ms", "0");              // do not wait to fill a batch
latency.put("batch.size", "16384");          // small 16 KB batches
latency.put("acks", "1");                    // leader-only; skip slow-replica wait
latency.put("compression.type", "none");     // skip CPU time on compression
latency.put("max.in.flight.requests.per.connection", "5");
// Result: low p99, modest peak throughput, weaker durability.

// Throughput-optimized producer: amortize everything.
Properties throughput = new Properties();
throughput.put("bootstrap.servers", "broker1:9092");
throughput.put("linger.ms", "50");           // wait up to 50 ms to fill a batch
throughput.put("batch.size", "262144");      // large 256 KB batches
throughput.put("acks", "all");               // full durability; accept the wait
throughput.put("compression.type", "lz4");   // fewer bytes, spend some CPU
throughput.put("max.in.flight.requests.per.connection", "5");
throughput.put("buffer.memory", "67108864"); // 64 MB to absorb bursts
// Result: very high throughput, higher p99, full durability.
```

Read down the two blocks and you can see the tradeoff at every line. The latency config waits for nothing, batches small, skips compression CPU, and accepts weaker durability to avoid the replica wait. The throughput config waits to fill fat batches, compresses to move fewer bytes, and accepts the durability wait of `acks=all`. There is no setting where one config is strictly better; each is the right answer for a different target, and the wrong answer for the other one. If you ever find yourself unsure which config a service should use, the question to ask is not "which is faster" — both are faster, on different axes — but "is a human or an upstream blocked on each message?" If yes, the left block; if no, the right block.

## 3. Little's Law and what it tells you

To reason about throughput and latency together — not as two separate dials but as a coupled system — you need one equation, and it is one of the most useful, most underused results in all of systems engineering. It is called **Little's Law**, and it states:

> **L = λ × W**

where **L** is the average number of items in the system, **λ** (lambda) is the average arrival rate (items per unit time, which in steady state equals the throughput), and **W** is the average time an item spends in the system (the latency). That is the whole law. Three quantities, one multiplication. It holds for any stable system in steady state, regardless of the distribution of arrivals, the service discipline, or the internal structure. It is almost magically general — it applies to a Kafka partition, a RabbitMQ queue, a thread pool, a checkout line at a grocery store, and the cars on a stretch of highway, all with the same three letters.

What makes it powerful for our purposes is that it ties throughput (λ) and latency (W) directly to a third quantity, the number of in-flight items (L), and lets you solve for any one given the other two. If you know two of the three, you know the third. That is an extraordinary amount of leverage for one multiplication.

#### Worked example: how many messages are in flight at 50,000 msg/s and 20 ms latency

Your message system is processing **λ = 50,000 messages per second** in steady state, and the average end-to-end time a message spends in the system — from produced to fully processed — is **W = 20 ms = 0.020 seconds**. How many messages are "in flight" at any instant — sitting in producer buffers, in the broker log not yet consumed, in fetch responses, or being processed?

Apply the law: **L = λ × W = 50,000 × 0.020 = 1,000 messages.** At any given instant, on average, a thousand messages are somewhere inside your system, not yet done. That number is not a guess; it falls straight out of the arithmetic. And it is immediately actionable. It tells you how big your in-flight buffers, your consumer prefetch, and your processing concurrency need to be sized to keep up. If your consumer prefetch buffer holds only 200 messages but the steady-state in-flight count is 1,000, you have a structural mismatch that will cause stalls. If you provisioned memory assuming 100 in-flight messages and the law says 1,000, you will run out of memory under load.

Now watch what the law tells you about the tradeoff. Suppose your throughput is fixed by the business — you must handle 50,000 msg/s, that is the load, you do not get to choose it. Little's Law says L = 50,000 × W. If your latency W doubles from 20 ms to 40 ms — say because a downstream service got slower — then L doubles to 2,000 in-flight messages. Higher latency at fixed throughput means *more stuff piled up inside the system*. That pile is your queue depth, your memory pressure, your lag. This is why a latency regression under constant load shows up as growing queues and growing memory — Little's Law guarantees it. Conversely, if you want to reduce the number of in-flight messages (to cut memory or lag) without dropping throughput, the *only* lever the law gives you is to reduce W — make each message faster through the system. You cannot wish L down; it is pinned by λ and W.

### Reading the law in three directions

The reason Little's Law is worth memorizing is that you can read it three ways depending on what is fixed and what you control:

- **Latency is rising and you do not know why.** If throughput λ is steady but in-flight L is climbing, then W must be climbing too (W = L / λ). Growing queues are a latency problem wearing a throughput costume. Look for the slow stage.
- **You want to hit a throughput target.** λ = L / W. To get more throughput at fixed per-message latency, you must increase L — more parallelism, more in-flight requests, deeper pipelines. If you cannot increase L (buffers are capped, concurrency is capped), your throughput is capped at L / W no matter how much load you offer.
- **You are sizing buffers and pools.** L = λ × W tells you the steady-state occupancy you must provision for. Under-provision L and the system stalls waiting for buffer space; over-provision and you waste memory and risk hiding a latency problem.

That third reading is the one that has saved me the most pain operationally. When someone says "the consumer keeps stalling under load," the first thing I compute is L = λ × W and compare it to the configured prefetch / in-flight limits. More often than not the configured limit is below the law's L, and the fix is simply to raise the buffer to match physics.

### The assumption that bites you: steady state

Little's Law has exactly one assumption, and it is the assumption people forget at the worst possible moment: it holds in **steady state** — when arrivals and departures are balanced over the averaging window, so that the system is neither filling up nor draining out on average. During a transient — a burst, a deploy, a downstream slowdown — the system is not in steady state, arrivals exceed departures, and L grows over time rather than holding constant. The law still holds *over a long enough window that includes the recovery*, but it does not give you a single static L during the transient itself. This matters because the transients are exactly when you care most. When a burst hits, L spikes above the steady-state value, and if your buffers were sized for the steady-state L, they overflow during the burst even though they are "correctly" sized for the average. The practical correction is to size buffers for the *peak* L you expect during a realistic burst, not the average L — which means you need to know your burst factor, the ratio of peak arrival rate to average arrival rate, and multiply the steady-state L by it. A system with a 3x burst factor needs buffers sized for roughly 3x the steady-state in-flight count, or it will stall every time a burst arrives, and stall precisely when load is highest and stalling hurts most.

There is a second subtlety worth internalizing: the W in Little's Law is the *total* time in the system, end to end, including every queue and every service stage. If you apply the law to a sub-system — say, just the broker, or just the consumer's processing stage — then λ, L, and W must all refer to that same sub-system. The law composes: you can apply it stage by stage down the pipeline, and the L for the whole system is the sum of the L's for each stage. That decomposition is enormously useful for diagnosis, because it lets you compute the in-flight count at each stage separately and find the one where the pile is building. If the broker's L is small but the consumer's processing L is huge, the pile is in processing, and that is where your latency lives. The law does not just give you one number; applied stage by stage, it gives you a map of where the work is sitting, and the stage holding the most work is almost always the stage you need to tune.

## 4. Utilization and the latency cliff

Little's Law tells you about steady state. The next piece of theory tells you what happens as you push a system toward its limit, and it is the single most important operational fact in this entire post: **latency does not degrade gracefully as you approach saturation. It explodes.**

Define **utilization**, usually written **ρ** (rho), as the fraction of capacity you are using: arrival rate divided by maximum service rate. ρ = 0.5 means you are running at half capacity; ρ = 0.95 means you are running at 95 percent of the maximum rate the system can sustain. Intuition — the dangerous, wrong intuition — says that latency should rise smoothly and proportionally as utilization climbs, so that 95 percent utilization gives you maybe twice the latency of 50 percent. Intuition is catastrophically wrong here.

Basic queueing theory gives the average wait in a simple queue as proportional to **ρ / (1 − ρ)**. Look at what that function does. At ρ = 0.5, it is 0.5/0.5 = 1. At ρ = 0.8, it is 0.8/0.2 = 4. At ρ = 0.9, it is 0.9/0.1 = 9. At ρ = 0.95, it is 0.95/0.05 = 19. At ρ = 0.99, it is 0.99/0.01 = 99. The wait is not rising linearly; it is rising toward a vertical asymptote at ρ = 1. Going from 90 to 95 percent utilization roughly *doubles* your queueing delay. Going from 95 to 99 percent multiplies it by five. As utilization approaches 100 percent, the denominator approaches zero and the latency goes to infinity. There is a cliff, and the cliff is real, and most production latency incidents are someone walking off it.

The before-and-after figure makes this concrete with two operating points. At 60 percent utilization the queue is short, the wait is a small multiple of the service time, and p99 latency sits stable at a low value. At 95 percent utilization the queue is deep, the wait is roughly 20x the service time, and p99 has spiked by more than an order of magnitude. The system is doing only slightly more work — 95 versus 60 percent — but the latency experience is completely different. That is the cliff, drawn.

![A before and after comparison showing a system at sixty percent utilization with a short queue and stable low p99 latency versus the same system at ninety-five percent utilization with a deep queue and p99 latency spiking by more than an order of magnitude](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-5.webp)

### Why the cliff exists

Think of it as a checkout line, since the math is identical. If customers arrive at exactly even intervals and each takes exactly the same time to serve, you could run at 100 percent utilization with no line at all — perfect pipelining. But real arrivals are bursty and real service times vary. When a burst arrives, the line grows. At low utilization, the line drains during the lulls between bursts, so it never grows without bound. At high utilization, there is no slack to drain in — the server is almost always busy — so each burst's backlog does not fully clear before the next burst piles on top of it. The backlogs stack. The closer to 100 percent you run, the less slack exists to absorb variance, and variance is what builds queues. At exactly 100 percent there is zero slack and any burst, however small, creates a backlog that never drains. The cliff is the price of variance meeting a lack of headroom.

This has a blunt operational consequence: **you must run brokers and consumers with headroom.** A broker pinned at 95 percent CPU or disk is not "efficiently utilized"; it is one traffic bump away from a latency cliff. The reason a small load increase can turn a healthy system into a melting one is precisely the ρ/(1−ρ) curve — near the asymptote, a 5 percent load increase is not a 5 percent latency increase, it is a multiplication. I have watched a service that ran fine at 88 percent utilization for months become completely unresponsive the day a marketing email pushed it to 96 percent. Nothing was broken. The system was simply operating in the steep part of the curve, where small inputs produce huge outputs, and the marketing email was the small input.

### The headroom rule of thumb

A practical rule that falls out of the curve: target somewhere around **60 to 70 percent utilization** for latency-sensitive systems, and you can push to 80 to 85 percent for throughput-sensitive systems where you care about cost efficiency more than tail latency and you can tolerate the occasional spike. Above 85 percent you are in cliff territory and you need either autoscaling that reacts before you get there or a hard load ceiling that sheds or delays excess work. The exact number depends on how bursty your traffic is — burstier traffic needs more headroom — but the shape of the rule is universal: the more you value low latency, the further from saturation you must run, and the gap you leave is not waste, it is the latency you are buying.

## 5. Tail latency: why the average lies

Up to now I have been a little loose, talking about "the latency" as if it were one number. It is not. Latency is a distribution, and the most important practical truth about that distribution is this: **the average latency lies, and the tail tells the truth.**

Consider a system whose latency is 5 ms for 99 percent of messages and 500 ms for the unlucky 1 percent. The average is 0.99 × 5 + 0.01 × 500 = 4.95 + 5 = 9.95 ms. Roughly 10 ms average. That sounds great. You would put "10 ms average latency" on a slide and feel good. But one message in a hundred takes half a second, and if your users send a hundred messages in a session, almost every user hits at least one of those 500 ms stalls. The average said 10 ms; the user experience is dominated by the 500 ms tail. The average hid the thing that actually matters.

This is why serious latency work is always done in **percentiles**, not averages. The p50 (median) is the latency that half of messages beat. The p99 is the latency that 99 percent of messages beat — only 1 in 100 is slower. The p999 (p99.9) is the latency that 99.9 percent beat — 1 in 1,000 is slower. The high percentiles are the **tail**, and the tail is where production pain lives, for a reason that is not obvious until you have been burned by it: **a single user request often fans out into many message operations, and the slowest one determines the user's experience.**

#### Worked example: how fan-out amplifies the tail

Suppose your p99 per-message latency is a respectable 10 ms, meaning any single message has a 99 percent chance of being faster than 10 ms, hence a 1 percent chance of being slower. Now suppose a user-facing request must produce-and-consume 100 messages and cannot complete until all 100 are done — a fan-out then join, extremely common in microservice architectures. What is the chance the *whole request* avoids the 10 ms tail? It is the chance all 100 messages are individually fast: 0.99 to the 100th power, which is about 0.366. So there is only a 37 percent chance the request escapes the tail entirely, and a **63 percent chance at least one of its 100 messages hits the slow path** and drags the whole request out to 10 ms or beyond. Your p99 was 10 ms per message, but at the request level — which is what the user feels — the tail dominates almost two requests in three. The per-message p99 looked fine and the user experience was terrible, because fan-out turns a rare per-message tail into a common per-request reality.

This is the mechanism behind a phenomenon Google's Jeff Dean named "tail at scale": the more a request fans out, the more the system's tail latency, not its median, governs the user experience. A 1-in-1,000 stall is invisible in isolation and inescapable at a fan-out of 1,000. It is why people who operate large systems obsess over p99 and p999 and treat the average as nearly worthless for latency-sensitive work.

### What creates the tail

The tail is not random noise; it has specific, recurring causes, and knowing them lets you hunt it:

- **Garbage collection pauses.** A JVM-based broker or client that stops the world for 200 ms turns every in-flight message into a tail event for the duration. Kafka and many clients are on the JVM; GC tuning is tail-latency tuning.
- **Queueing under bursts.** From the previous section: when a burst pushes utilization briefly toward 1, the messages caught in that burst wait far longer than the median. The cliff produces the tail.
- **Retries and timeouts.** A message that hits a transient broker error and retries with backoff has a latency that is the sum of the timeout plus the retry — easily 10 to 100x the median. A handful of retries create the p999.
- **`fsync` and disk stalls.** A broker flushing to disk can stall when the page cache is dirty and the disk is busy. The flushes that catch a busy disk are the slow ones.
- **Head-of-line blocking.** One slow message at the front of a partition or a single-threaded consumer blocks every message behind it, even fast ones. The blocked ones inherit the slow one's latency.
- **Rebalances and leader elections.** A consumer-group rebalance or a partition leader change pauses progress for hundreds of milliseconds to seconds; everything in flight during the pause is a tail event.

Every one of these is a place where, in pursuit of throughput, you can accidentally inflate the tail. Bigger batches mean a GC pause stalls more in-flight messages at once. Higher utilization means more burst queueing. Aggressive retry-for-durability means more retry tails. The throughput-vs-latency tradeoff, viewed through the tail, is even sharper than through the average: chasing throughput often improves the median while *quietly destroying the tail*, and since the tail is what users feel, you can "improve performance" by every average metric and make the system feel worse.

### Why batching specifically inflates the tail

It is worth dwelling on the interaction between batching and the tail, because it is the most common way a well-intentioned throughput tuning silently regresses the experience. When you raise the batch size and linger to lift throughput, you change the shape of the latency distribution in two ways at once. First, you raise the median slightly, because every message now waits a bit longer for its batch — that is the obvious, expected cost. Second, and far less obvious, you fatten the tail, because a batch is now a unit of *correlated failure*. If a batch hits a retry, every message in that batch retries together; a bigger batch means more messages share each retry tail. If a GC pause or a disk stall lands while a large batch is in flight, more messages are caught by it. The tail events do not become more frequent, but each one now captures more messages, so the fraction of messages experiencing a tail event grows even though the per-event probability is unchanged. This is why a change that looks like a pure throughput win in the average and even the p99 can show up as a p999 regression: the rare bad events now sweep up more victims each. The discipline is to always check the p999, not just the p50 and p99, after any batching change, because the p999 is where this correlated-failure effect first becomes visible, and it is exactly the percentile a fan-out workload feels.

### Tail latency and the durability axis

There is a third axis lurking under every throughput-vs-latency discussion, and it touches the tail directly: **durability**. The `acks` setting is nominally a durability knob, but it expresses itself as latency, and specifically as *tail* latency. With `acks=all`, a produce request is not acknowledged until every in-sync replica has the data. The latency of that acknowledgement is the latency of the *slowest* replica, not the average one — and the slowest replica is, by definition, a tail event on the replication side. So `acks=all` couples your produce tail latency to the replication tail latency of your slowest follower. A single slow replica — one with a busy disk, a GC pause, or a saturated network link — drags the produce p99 of every partition it follows. This is why durability is not free even when you have the throughput to spare: it imports the tail of your slowest replica into your produce path. Teams that run `acks=all` and then wonder why their produce p999 is bad are usually looking at one sick broker dragging the whole cluster's tail, and the fix is to find and heal the slow replica, not to touch any producer knob. The three axes — throughput, latency, durability — are genuinely three-way coupled, and the coupling shows up most sharply in the tail.

## 6. The knobs and which axis each moves

Now that the theory is in place, let us catalogue the actual knobs you will turn in a real message system and, for each, which way it bends the curve. The stack figure shows where latency physically accumulates as a message travels: produce and serialize, then wait in the batch, then replicate, then wait in the fetch, then process. Each layer is a place with at least one knob, and the total latency is the sum down the stack. Reducing the total means attacking the fattest layer, which is usually the batch wait or the processing — not, as people often assume, the network.

![A layered stack showing where end to end latency accumulates from produce and serialize through batch wait through replication through fetch wait through consumer processing with the total being the sum of all layers](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-4.webp)

### Producer-side knobs

These are covered in depth in the [producer optimization](/blog/software-development/message-queue/producer-optimization-batching-compression-acks) post; here is how each moves the two axes.

- **`batch.size`** (byte cap on a batch). Larger → higher throughput (better amortization), and higher latency only at low load where batches fill slowly. At high load it is nearly free. Direction: throughput.
- **`linger.ms`** (time to wait for a batch to fill). Higher → higher throughput, higher latency, with the latency cost concentrated at low load as we computed. Direction: throughput, paid in latency.
- **`acks`** (durability acknowledgement level). `acks=0` (fire and forget) → lowest latency, highest throughput, zero durability. `acks=1` (leader only) → low latency, can lose data on leader failure. `acks=all` (all in-sync replicas) → highest durability, highest latency (wait for slowest replica). This one trades a *third* axis — durability — but it shows up as latency: more durable means waiting for more confirmations.
- **`compression.type`** (none / lz4 / zstd / gzip / snappy). Compression → higher *effective* throughput (fewer bytes over the wire and to disk), at the cost of CPU time, which is latency. lz4 and zstd are the usual sweet spots — strong ratio, cheap CPU. gzip compresses hardest but is slowest; it is a throughput-at-the-cost-of-latency choice.
- **`max.in.flight.requests.per.connection`** (how many unacknowledged requests can be outstanding). Higher → more throughput via pipelining (more requests in flight, more L in Little's Law), at some risk to ordering on retries. Direction: throughput.
- **`buffer.memory`** (total producer buffer). Larger → absorbs bigger bursts without blocking the producer thread, raising sustained throughput. But a deep buffer that fills under sustained overload adds queueing latency — messages sit in the buffer. Direction: throughput, with a latency tail under overload.

### Broker-side knobs

- **Replication factor and `min.insync.replicas`.** More replicas → more durability and read availability, more latency under `acks=all` (more confirmations to wait for). Direction: durability/availability, paid in latency.
- **`num.io.threads` / `num.network.threads`.** More threads → more parallelism, more throughput, up to the point where they saturate CPU or disk. Direction: throughput.
- **Page cache and `fsync` policy.** Relying on the OS page cache and flushing lazily → high throughput, low latency, at the cost of durability if the machine loses power before flush. Flushing on every write → durable but slow. Direction: a throughput/durability trade.

### Consumer-side knobs

These get the full treatment in the [consumer optimization](/blog/software-development/message-queue/consumer-optimization-and-scaling) post.

- **`fetch.min.bytes`** (minimum bytes the broker waits to accumulate before answering a fetch). Higher → fewer, fatter fetches, more throughput, but the broker holds the response (up to `fetch.max.wait.ms`), which is latency. This is the consumer-side twin of producer linger. Direction: throughput, paid in latency.
- **`fetch.max.wait.ms`** (how long the broker waits for `fetch.min.bytes` before answering anyway). Higher → more batching on fetch, more latency. Direction: throughput.
- **`max.poll.records`** (how many records one `poll()` returns). Higher → more work per poll, better amortization of the poll loop, at the risk of longer processing time per poll and rebalance pressure if processing overruns `max.poll.interval.ms`. Direction: throughput.
- **Consumer parallelism (partitions and consumer instances).** More partitions and more consumer instances → more throughput via parallel processing. This is the cleanest throughput lever that does *not* cost latency — adding a parallel consumer to drain a queue faster reduces queue depth and therefore reduces latency, by Little's Law. Parallelism is the rare knob that can help both axes, up to the point where you hit a shared bottleneck (the broker, a database, a downstream service).

That last point deserves emphasis because it is the most useful exception to the tradeoff. **Parallelism is the one lever that can buy throughput without spending latency** — sometimes it buys both. If your latency is high because a queue is backed up, adding consumers drains the queue, drops the in-flight count L, and by Little's Law drops the per-message time W. The tradeoff is not violated; you have simply raised the system's service rate so utilization drops off the cliff. The catch is that parallelism only helps until you hit a downstream bottleneck — ten consumers all hammering one database do not go faster than the database, and then you are back to spending latency for throughput at the new bottleneck. Parallelism moves the bottleneck; it does not abolish it.

### The same tradeoff in RabbitMQ: prefetch

The knobs are named differently in RabbitMQ but the tradeoff is identical, which is the whole point of learning it as a principle rather than a config list. RabbitMQ's consumer-side equivalent of the fetch/batch knob is **prefetch** (the `basic.qos` prefetch count), which controls how many unacknowledged messages the broker will push to a consumer before waiting for acks. A low prefetch — say 1 — means the broker sends one message, waits for the ack, then sends the next. That minimizes the number of in-flight messages and keeps any single message from waiting behind a backlog at the consumer, but it caps throughput hard, because every message costs a full round-trip of broker-to-consumer-to-ack before the next moves, and at a 1 ms round-trip you are limited to roughly 1,000 messages per second per consumer no matter how fast the consumer processes. A high prefetch — say 250 — lets the broker stream a couple hundred messages ahead, so the consumer always has work and the round-trips overlap, which lifts throughput enormously. The cost is exactly the cost we have been discussing: those 250 prefetched messages are now in flight at the consumer, and if the consumer is slow, a message at the back of that local buffer waits behind 249 others — added latency, and a fatter unit of correlated loss if the consumer crashes before acking. This is producer linger and Kafka `fetch.min.bytes` in a different costume. The right prefetch is small for latency-sensitive consumers and large for throughput-sensitive ones, and the [RabbitMQ production architecture](/blog/software-development/system-design/rabbitmq-production-architecture-scaling) post goes deep on the exact numbers. The lesson here is that the tradeoff is universal across brokers; only the spelling of the knobs changes.

```python
# RabbitMQ prefetch: the same throughput-vs-latency knob, AMQP spelling.
import pika

connection = pika.BlockingConnection(pika.ConnectionParameters("rabbit1"))
channel = connection.channel()

# Latency-sensitive: prefetch=1. One in-flight message, no local backlog,
# but throughput capped at ~1/RTT messages per second.
channel.basic_qos(prefetch_count=1)

# Throughput-sensitive: prefetch=250. The broker streams ahead so the
# consumer never starves, but up to 250 messages sit in flight per consumer.
channel.basic_qos(prefetch_count=250)
```

## 7. Picking a target: latency, throughput, or balanced

The single most important act in tuning a message system is not turning any particular knob. It is deciding, before you touch anything, which of the three targets you are tuning toward. Everything downstream is mechanical once the target is set; everything is thrashing if it is not. The tree figure lays out the taxonomy: every tuning effort serves a latency-sensitive goal, a throughput-sensitive goal, or a balanced one, and each goal dictates a coherent set of knob settings.

![A taxonomy tree of tuning goals branching from a single root into latency-sensitive throughput-sensitive and balanced each leading to its characteristic linger batch and target settings](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-7.webp)

### Latency-sensitive

You are latency-sensitive when something — a user, an upstream request, a trading decision, a control loop — is blocked waiting on each individual message, and the value of the message decays fast with delay. Synchronous request-reply over messaging, real-time bidding, fraud-scoring in the checkout path, control systems: these are latency-sensitive. Here the correct settings push everything toward immediacy: `linger.ms=0`, small batches, `acks=1` if you can tolerate the durability risk (or `acks=all` with fast, co-located replicas if you cannot), `fetch.min.bytes=1`, low `fetch.max.wait.ms`, generous headroom so you stay far from the cliff (target 60 to 70 percent utilization), and aggressive monitoring of p99 and p999, not the average. You accept lower peak throughput and higher cost-per-message in exchange for each message arriving soon. You run more, smaller, less-loaded instances rather than fewer, fuller ones.

### Throughput-sensitive

You are throughput-sensitive when the goal is to move a large volume of data and no individual message is on anybody's critical path. ETL and bulk ingest, log and metrics aggregation, data-lake loading, replication and CDC streams, backfills, batch analytics: these are throughput-sensitive. Here you push everything toward amortization: large `batch.size` (256 KB to 1 MB+), generous `linger.ms` (25 to 100 ms), `compression.type=lz4` or `zstd`, high `fetch.min.bytes` and `fetch.max.wait.ms`, large `max.poll.records`, deep buffers, and you can run hot — 80 to 85 percent utilization — because you care about cost efficiency and a latency spike on a backfill bothers no one. You accept that any single message might take tens or hundreds of milliseconds in exchange for moving a mountain of them per dollar.

### Balanced

Most general-purpose event-driven systems are balanced. They are not on a human's critical path, so a few tens of milliseconds is fine, but they are not pure bulk either, so you do not want to wait hundreds of milliseconds. Order events, inventory updates, notifications, most domain events in a microservice mesh: balanced. Here you pick moderate settings — `linger.ms=5`, medium batches, `acks=all` for durability (these systems usually care about not losing events), `fetch.min.bytes` of a few kilobytes, utilization targets around 70 to 75 percent — and you tune by watching whether latency or throughput becomes the binding constraint in practice, then nudge toward whichever is hurting. The balanced target is not a cop-out; it is the honest answer for a system that genuinely values both and optimizes neither to the extreme.

### Do not mix targets on one topic

A trap worth naming: do not try to serve latency-sensitive and throughput-sensitive traffic through the same topic with the same client config, because the config can only point one way. If your checkout-critical events and your analytics firehose share a topic and a producer, you will either starve the checkout events of latency (if you tune for the firehose's throughput) or starve the firehose of throughput (if you tune for checkout's latency). The right move is to separate them — different topics, often different clusters — so each can be tuned toward its own target. A single config cannot straddle two targets; the tradeoff forbids it.

## 8. Measuring it: percentiles, load tests, saturation

You cannot tune what you cannot measure, and message-system performance is measured wrong constantly. Here is how to measure it right.

### Measure percentiles, never just averages

From section 5, the average lies. Your dashboards must show p50, p99, and p999 latency, not just the mean. The mean is acceptable as a coarse health signal, but every latency SLO and every tuning decision is made on percentiles. A subtle but critical point: **you cannot average percentiles across machines or time windows.** The p99 of two hosts is not the average of their two p99s; percentiles do not compose that way. To get a true global p99 you need to aggregate the underlying latency histograms (HDR histograms, t-digests, or the broker's own latency metrics) and compute the percentile from the merged distribution. Averaging p99s across hosts systematically *understates* your true tail, which means you think you are doing better than you are precisely at the part of the distribution that matters most.

### Load-test toward saturation, not just at expected load

The single most common load-testing mistake is testing only at expected load and declaring victory. Expected load tells you the system works today; it tells you nothing about how close to the cliff you are or what happens when load doubles. You must **push the system toward saturation** in a load test — keep increasing the offered rate and watch where the latency curve turns vertical. That inflection point is your real capacity, and it is almost always lower than the throughput number on the broker's spec sheet, because the spec sheet number is the saturation throughput at infinite latency, which is operationally useless. Your usable capacity is the throughput at which p99 latency is still acceptable, and that point is well below saturation. Plot offered load on one axis and p99 latency on the other; the usable capacity is where p99 crosses your SLO, not where the system finally falls over.

```bash
# Kafka's built-in producer perf test: push throughput and watch latency.
# Sweep --throughput from low to -1 (unbounded) to find the saturation point.
kafka-producer-perf-test.sh \
  --topic perf-test \
  --num-records 10000000 \
  --record-size 1024 \
  --throughput 200000 \
  --producer-props \
      bootstrap.servers=broker1:9092 \
      acks=all \
      linger.ms=5 \
      batch.size=131072 \
      compression.type=lz4
# Output reports records/sec AND latency percentiles:
#   ... 198431 records/sec, ... 50th 4 ms, 95th 11 ms, 99th 23 ms, 99.9th 88 ms
# Re-run with --throughput 400000, 600000, ... and watch the 99.9th column
# climb non-linearly. The rate where p99.9 crosses your SLO is your real capacity.
```

The thing to watch in that sweep is not the records/sec — that will keep climbing until the broker saturates. It is the p99 and p999 columns. They sit roughly flat through the healthy range and then bend sharply upward as you approach saturation. The bend is the cliff from section 4, measured directly. Your capacity is the offered rate just before the bend, with margin.

### Measure under realistic conditions

A load test with uniform message sizes, perfectly even arrival rates, and no competing traffic will give you optimistic numbers, because it has none of the variance that builds queues. Real traffic is bursty, mixed in size, and contends with replication, other consumers, and GC. Test with realistic burstiness (replay a production traffic capture if you can), realistic message-size distribution, and the real durability settings (`acks=all` if that is what production uses — testing with `acks=1` and shipping `acks=all` is a classic way to be surprised by latency in production). The closer your test's variance is to production's variance, the closer your measured cliff is to the real one.

### Watch utilization and lag as leading indicators

Latency is a lagging indicator — by the time p99 spikes, you are already on the cliff. The leading indicators are **utilization** (broker CPU, disk, network; consumer thread busy fraction) and **consumer lag** (how far behind the consumers are). Rising utilization toward 85 percent is a warning that you are approaching the steep part of the curve before latency visibly degrades. Rising lag is Little's Law in action — L climbing at fixed λ means W is climbing, latency is degrading, and the queue is building. Alert on utilization and lag *trends*, not just on latency thresholds, so you get warning before the cliff instead of after.

## 9. A tuning methodology

Putting it together, here is the methodology I use, in order. It is deliberately boring, because effective tuning is boring — it is measurement and small deliberate changes, not heroics.

**Step 1: Declare the target.** Latency-sensitive, throughput-sensitive, or balanced. Write it down. Get agreement. Every later decision references this. If you skip this step you will tune in circles, because without a target there is no definition of "better" — every change improves one axis and worsens the other, and with no target you cannot say which trade is good. The pipeline figure shows what a declared latency target looks like in practice: a total budget divided across the hops, so each stage has a number to hit and you can see immediately which stage is over budget.

![A pipeline showing an end to end latency budget of twenty-five milliseconds divided across produce linger replicate fetch and process stages each with its own allocated milliseconds summing to the target](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-6.webp)

**Step 2: Establish a baseline.** Measure current throughput and the full latency distribution (p50, p99, p999) under realistic load before changing anything. You cannot tell if a change helped without a before. Most "this made it faster" claims evaporate when someone asks for the baseline, because there wasn't one.

**Step 3: Find the binding constraint.** Where is the time actually going? Use the stack from section 6: is the latency in produce, batch wait, replication, fetch wait, or processing? Profile it. The fattest stage is your target; tuning anything else is wasted effort. Apply Little's Law: compute L = λ × W and check whether buffers are sized for it. Almost always the binding constraint is one specific stage, and almost always it is not the one people assume — it is usually processing or batch wait, rarely the network.

**Step 4: Change one knob, measure, keep or revert.** Single-variable changes only. Turn one knob in the direction your target dictates, re-run the load test, compare against baseline on *both* axes (because every knob moves both). Keep the change only if it improved your target axis more than it hurt the other one by an amount you find acceptable. Then move to the next knob. Changing five knobs at once and seeing the number improve teaches you nothing about which knob did it, and worse, leaves you unable to back out the one that quietly wrecked your tail.

**Step 5: Respect the cliff.** Whatever target you chose, confirm where saturation is and set your operating point with headroom below it. For latency-sensitive, 60 to 70 percent; for throughput-sensitive, 80 to 85 percent. Set autoscaling or load-shedding to keep you off the cliff, not to let you wander onto it.

**Step 6: Watch the tail in production.** The load test approximates; production is the real distribution. Keep p99 and p999 on the dashboard, alert on utilization and lag trends as leading indicators, and revisit when traffic shape changes. A config tuned for last quarter's traffic can be wrong for this quarter's, because the tradeoff's right answer depends on the load, and the load moves.

The before-and-after below contrasts the two ways this methodology can go. On the left, the team chased throughput, ran at 95 percent utilization, watched the average and called it fine — while the p999 quietly blew through the SLO. On the right, the team designed for bounded latency, capped load at 70 percent to keep headroom for bursts, and kept the p999 inside the SLO at the cost of running more instances. Same workload, same broker. The difference is that one team optimized the number on the slide (the average) and the other optimized the number the users feel (the tail), and only one of them met the SLO.

![A before and after comparison showing a team chasing throughput at ninety-five percent utilization whose average looks fine but whose p999 breaches the SLO versus a bounded-latency design capping load at seventy percent with headroom that keeps p999 inside the SLO](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-9.webp)

## Case studies and war stories

### The linger that was set in production by a copy-paste

A team I worked with had copied a Kafka producer config from a blog post optimized for a data-lake ingest pipeline. The config had `linger.ms=100` and `batch.size=1048576` — one megabyte batches, a hundred milliseconds of linger. Perfect for a throughput-sensitive backfill. They pasted it into a service that produced order-confirmation events in the synchronous checkout path. Every checkout now ate up to 100 ms of pure linger waiting for a batch that, at their modest order rate, never came close to filling. Customers experienced a checkout that felt sluggish for no visible reason. The fix was a one-line change to `linger.ms=0` and a small batch size, and checkout latency dropped by roughly 90 ms at p99 overnight. The lesson is the one from section 7: a config is only correct relative to a target, and a config copied from a system with a different target is not a starting point, it is a bug. The throughput-tuned config was not "wrong" — it was right, for a workload they did not have.

### The broker that ran at 92 percent and was "fine" until it wasn't

A streaming platform ran its Kafka brokers at a steady 92 percent disk utilization. Capacity planning had concluded this was efficient — why pay for disks you are not using? For months it was fine: p99 produce latency sat around 15 ms. Then a downstream outage caused a fleet of consumers to disconnect and reconnect, and the resulting fetch storm pushed disk utilization to 99 percent for about ninety seconds. In those ninety seconds, produce p99 went from 15 ms to over 2 seconds, producers' buffers filled, `send()` calls started blocking, and the upstream services — which produced synchronously — began timing out and cascading. A ninety-second disk spike turned into a fifteen-minute multi-service outage. The post-incident fix was to add disk headroom and target 75 percent steady-state utilization. The lesson is section 4 in the flesh: 92 percent was not "efficient," it was sitting in the steep part of the ρ/(1−ρ) curve, where a small, transient load increase is not a small latency increase but a multiplicative one. The headroom they had been "saving" was the latency stability they did not have.

### The average that hid the GC pause

A payments service reported a healthy 8 ms average consume-to-process latency and a green dashboard for months. But support kept getting sporadic complaints about payments that "hung for a second." Nobody could reproduce it, because on average everything was fast. When the team finally added p999 to the dashboard, the truth appeared: p50 was 4 ms, p99 was 18 ms, and p999 was 1,100 ms. One payment in a thousand was taking over a second. The cause was a JVM garbage-collection pause on the consumer that stopped the world for about a second every few minutes; the messages caught in each pause were the p999. The average — and even the p99 — never showed it, because a 1-in-1,000 event barely moves either. The fix was GC tuning (switching collectors and shrinking the heap) which cut the p999 to 60 ms. The lesson is section 5: the average and even the p99 can be healthy while the p999 is on fire, and the p999 is exactly what a user with bad luck experiences. If you do not measure the tail, you do not know your latency; you know a comforting story about it.

### The parallelism that fixed latency by fixing throughput

A notification service had climbing latency — events were taking longer and longer from produced to delivered, creeping from 50 ms to several seconds over a week as traffic grew. The team's instinct was to tune for latency: lower linger, smaller batches. None of it helped, because the latency was not in the producer. Applying Little's Law diagnosed it instantly: throughput λ had grown, the single consumer instance's service rate was now below λ, so the queue depth L was growing without bound, and W = L / λ was therefore climbing. The latency was a *throughput* problem wearing a latency costume. The fix was not a latency knob at all — it was adding partitions and consumer instances to raise the service rate above the arrival rate, which dropped utilization off the cliff, drained the queue, and brought latency back to 50 ms. This is section 6's exception in action: parallelism was the one lever that bought throughput *and* latency at once, because the latency was caused by saturation, and the cure for saturation is more service capacity, not faster individual messages.

## When to reach for each target (and when not to)

The decision of which axis to optimize is not a matter of taste; it follows from the workload. The matrix below maps common workload types to their correct optimization target, linger setting, and acceptable tail. Use it as a starting point, not gospel — your specifics may shift a row — but the shape is reliable: interactive workloads optimize latency, bulk and aggregation workloads optimize throughput, and stream processing usually sits balanced.

![A matrix mapping workload types interactive RPC stream processing bulk ingest and log aggregation against the axis to optimize the linger setting and the acceptable p99 showing interactive favoring latency and bulk favoring throughput](/imgs/blogs/throughput-vs-latency-tuning-tradeoff-8.webp)

**Reach for latency optimization when** a human or a synchronous upstream is blocked on each message, when the value of a message decays in milliseconds (fraud, bidding, trading, control loops), or when you fan out a user request into many message operations and the tail will dominate. Accept lower throughput and higher cost per message. Run with generous headroom. Watch p99 and p999.

**Reach for throughput optimization when** you are moving bulk data — ETL, backfills, log and metrics aggregation, CDC, data-lake loading — and no individual message is on anyone's critical path. Accept that a single message may take tens or hundreds of milliseconds. Batch hard, compress, run hot, optimize cost per message. Watch throughput and cost; the tail does not matter much here.

**Reach for balanced when** the system genuinely values both and optimizes neither — most general event-driven systems. Pick moderate settings, keep `acks=all` for durability, target 70 to 75 percent utilization, and let production tell you which axis is the binding constraint, then nudge.

**Do not** mix latency-sensitive and throughput-sensitive traffic on one topic with one config — separate them so each can be tuned to its target. **Do not** run any latency-sensitive system above 85 percent utilization. **Do not** tune by averages. **Do not** copy a config from a system whose target differs from yours. **Do not** change multiple knobs at once and credit the win to the wrong one. And **do not** believe you can have maximum throughput and minimum latency simultaneously — the tradeoff is mechanical, and the engineers who accept it and pick a target beat the ones who keep trying to escape it.

## Key takeaways

- **Throughput is a rate; latency is a duration; they trade against each other** because the techniques that raise throughput work by amortizing fixed overhead across batched work, and batching requires waiting, and waiting is latency.
- **Batching is the canonical knob.** Linger's latency cost is highest exactly when its throughput benefit is lowest (low load) and nearly free when its benefit is highest (high load, where batches fill on size before the timer fires).
- **Little's Law (L = λ × W)** ties throughput, latency, and in-flight count into one equation. Use it to size buffers, to diagnose growing queues as a latency problem, and to know that at fixed throughput, rising latency means a rising pile of in-flight work.
- **Latency does not degrade linearly toward saturation — it explodes.** Queueing delay scales as ρ/(1−ρ), so going from 90 to 95 percent utilization roughly doubles delay and 95 to 99 percent quintuples it. Run with headroom; the gap you leave is the latency stability you buy.
- **The average lies; the tail tells the truth.** Measure p99 and p999, never just the mean, because fan-out turns a rare per-message tail into a common per-request reality, and the tail is what users feel.
- **Every knob moves both axes.** Batch size, linger, acks, compression, in-flight requests, fetch settings, buffers — each trades latency for throughput or back. The lone exception is parallelism, which can buy throughput and latency together by dropping utilization off the cliff, until it hits a downstream bottleneck.
- **Pick a target first — latency-sensitive, throughput-sensitive, or balanced — and tune toward it.** A config is only correct relative to a target; the same setting is a win for one workload and a bug for another.
- **Measure toward saturation, not just at expected load,** to find where the cliff is. Watch utilization and lag as leading indicators; latency is a lagging one.
- **You cannot maximize both.** Accept the tradeoff, declare your target, and tune deliberately. The whole craft is choosing where on the curve to sit and moving there on purpose.

## Further reading

- [Anatomy of a message system: producers, brokers, consumers](/blog/software-development/message-queue/anatomy-of-a-message-system-producers-brokers-consumers) — the vocabulary this post assumes.
- [What is a message queue: async decoupling and load leveling](/blog/software-development/message-queue/message-queues-async-decoupling-and-load-leveling) — why messaging exists and how load leveling relates to the cliff.
- [Producer optimization: batching, compression, acks](/blog/software-development/message-queue/producer-optimization-batching-compression-acks) — the producer-side knobs in depth.
- [Consumer optimization and scaling](/blog/software-development/message-queue/consumer-optimization-and-scaling) — the consumer-side knobs and how parallelism buys both axes.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — how the log structure underpins Kafka's throughput.
- "The Tail at Scale," Jeff Dean and Luiz André Barroso, Communications of the ACM, 2013 — the canonical paper on why fan-out makes the tail dominate.
- "Little's Law as Viewed on Its 50th Anniversary," John D. C. Little, Operations Research, 2011 — the law from the source.
- Kafka producer and consumer configuration reference (Apache Kafka documentation) — the authoritative list of every knob named here.
