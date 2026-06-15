---
title: "Back-of-the-Envelope Estimation for System Design: Fast, Defensible Numbers"
date: "2026-06-15"
publishDate: "2026-06-15"
description: "Learn the handful of numbers every engineer should memorize and the five-step workflow that turns daily active users into QPS, storage, bandwidth, servers, and a monthly dollar bill you can defend in a design review."
tags:
  [
    "system-design",
    "capacity-planning",
    "estimation",
    "architecture",
    "distributed-systems",
    "scalability",
    "performance",
    "cost-optimization",
    "back-of-the-envelope",
  ]
category: "software-development"
subcategory: "System Design"
author: "Hiep Tran"
featured: true
readTime: 52
image: "/imgs/blogs/back-of-the-envelope-estimation-for-system-design-1.webp"
---

The most useful skill in a system design review is not knowing the trendy database. It is being able to say, out loud, within ten seconds: "That's about 60,000 requests per second at peak, roughly 4 terabytes a year, and the hot set fits in RAM on a single box, so we don't need to shard yet." A senior engineer who can produce those numbers on demand reshapes the entire conversation. The people who can't tend to argue about technology choices in a vacuum, picking Cassandra because it "scales" without ever checking whether one Postgres instance would have handled the load for the next three years.

Back-of-the-envelope estimation is the antidote. It is the discipline of getting to within a factor of two or three of the right answer using arithmetic you can do in your head, a small set of memorized constants, and a workflow that always runs in the same order. You will be wrong in the second significant digit. That is fine. The decisions estimation drives — fits-in-RAM versus needs-sharding, one region versus three, cache-and-CDN versus brute-force reads — almost never hinge on the second digit. They hinge on the order of magnitude, and order of magnitude is exactly what an envelope calculation nails.

This post is the senior's estimation toolkit. We will start with the numbers worth memorizing — Jeff Dean's latency ladder and a few throughput constants — then build the workflow that chains them: from daily active users to queries per second, to storage per day and per year, to bandwidth, to server count, to cache memory, and finally to a rough monthly bill. We will run that workflow end to end on three worked examples — a URL shortener, a photo-sharing feed, and a chat system — showing every line of arithmetic. Along the way we will hammer the two ideas that separate a defensible estimate from a naive one: **peak versus average traffic**, and the **read-to-write ratio** that quietly determines your entire architecture. By the end you should be able to size a system from a one-line prompt and explain, in dollars and physical limits, why your design is the shape it is.

![A matrix of latency numbers every engineer should memorize spanning L1 cache through cross-region round trips, with relative cost columns showing six orders of magnitude](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-1.webp)

## Why estimate at all

Estimation earns its keep in three concrete situations, and it is worth being explicit about them because they tell you when the skill matters.

The first is **the design review**. Someone proposes an architecture. The question that should immediately follow is "what scale are we designing for?" If the proposer can answer in numbers, the review can interrogate the design against those numbers. If they can't, everyone is just trading opinions. A senior uses estimation to convert vibes into a falsifiable claim: "this design holds to 100k QPS; past that the database is the bottleneck." Now the review has something to test.

The second is **the early architecture decision**, made before any code exists. Sharding a database, introducing a message queue, standing up a CDN, going multi-region — these are expensive, hard-to-reverse commitments. Estimation tells you whether you need them *yet*. The single most valuable output of an envelope calculation is often a negative result: "we do not need to shard, the whole dataset is 400 GB and fits on one machine with room to spare." That negative result saves months of complexity. Premature distribution is one of the most common and most expensive mistakes in our industry, and a five-minute estimate prevents most of it.

The third is **cost and capacity planning**. Before you provision, you want to know roughly what the monthly bill will be and where it concentrates. Estimation lets you spot, in advance, that egress bandwidth is going to cost three times what the servers do, or that your storage grows unbounded and will dominate the bill by year two. Finding that on the whiteboard is free. Finding it on the invoice is a budget meeting.

The thread connecting all three is **speed**. Estimation has to be fast — a few minutes, mostly mental arithmetic — or you won't do it in the moments that matter. That speed is bought with two things: a set of memorized numbers, and aggressive rounding. We will get the numbers first.

## The numbers every engineer should memorize

There is a famous list, attributed to Jeff Dean, called "Latency Numbers Every Programmer Should Know." It is famous because it is genuinely load-bearing: once these numbers live in your head, an enormous range of design questions answer themselves. The exact figures drift with hardware generations, but the *ratios* between them are stable, and the ratios are what you reason with. Figure 1 above lays out the ladder. Here it is as a table you can drill until it's automatic.

| Operation | Latency | In human terms (×1 billion) |
| --- | --- | --- |
| L1 cache reference | ~1 ns | 1 second |
| Branch mispredict | ~3 ns | 3 seconds |
| L2 cache reference | ~4 ns | 4 seconds |
| Mutex lock/unlock | ~17 ns | 17 seconds |
| Main memory reference | ~100 ns | ~1.5 minutes |
| Compress 1 KB (fast) | ~2 µs | ~30 minutes |
| Read 1 MB sequentially from RAM | ~3 µs | ~50 minutes |
| SSD random read | ~16 µs | ~4.5 hours |
| Round trip within a datacenter | ~0.5 ms | ~6 days |
| Read 1 MB sequentially from SSD | ~50 µs | ~14 hours |
| Read 1 MB sequentially from disk (HDD) | ~1–2 ms | ~3 weeks |
| Disk seek (HDD) | ~10 ms | ~4 months |
| Round trip CA → Netherlands → CA | ~150 ms | ~5 years |

The "human terms" column multiplies every latency by a billion, which is a trick worth keeping. It converts nanoseconds and milliseconds — quantities your intuition has no feel for — into seconds, days, and years, which your intuition handles fine. In those terms, an L1 cache hit is a single heartbeat, a main-memory reference is grabbing a coffee, a datacenter round trip is a week-long road trip, and a cross-continent round trip is a five-year prison sentence. When you internalize that a cross-region call is *five years* relative to a cache hit's *one second*, you stop casually making cross-region calls in a hot loop.

A few derived rules fall straight out of this ladder, and these are the ones you actually use:

- **Memory is roughly 100× faster than SSD, and SSD is roughly 100× faster than a spinning disk seek.** Two big cliffs. Crossing either one in your hot path is a design decision, not an accident.
- **A datacenter round trip (~0.5 ms) is about 1,000× a memory reference but only ~1/300th of a cross-region round trip.** So fanning out to ten services *within* a region is usually fine; doing the same across regions is usually not.
- **Sequential beats random by a wide margin on every medium.** Reading 1 MB sequentially from SSD (~50 µs) is far cheaper per byte than a thousand random 1 KB reads. This is the entire reason log-structured storage exists — see the deep-dives on [B-trees and how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) and [LSM-trees as write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) for why sequential I/O dominates storage-engine design.

Now the throughput constants. Latency tells you how long one operation takes; throughput tells you how many you can do per second. These are rougher and more workload-dependent, but you need a default for each so you have something to divide by.

| Resource | Rough capacity | Notes |
| --- | --- | --- |
| Single commodity server, simple request | ~10k–50k QPS | Cheap reads/JSON; less if CPU-heavy |
| Single server, DB-backed request | ~1k–10k QPS | Bounded by the database, not the app |
| One disk (HDD), sequential | ~100–200 MB/s | Random is 100× worse |
| One SSD, sequential | ~500 MB/s–2 GB/s | NVMe at the top end |
| One NIC | ~1.25 GB/s (10 Gbps) | 10 GbE is the common baseline |
| Single Postgres/MySQL instance, writes | ~5k–15k writes/s | Higher with batching, tuning |
| Single Redis instance | ~100k+ ops/s | In-memory, single-threaded core |
| One Kafka broker | ~hundreds of MB/s, ~100k+ msgs/s | Sequential log append; see below |

The Kafka and Redis numbers connect to the mechanism deep-dives: Kafka's throughput comes from treating storage as an append-only sequential log, covered in [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log), and Redis's comes from staying in memory. The point of the table is not precision — your real numbers will differ by 2–5× depending on payload size, language, and tuning — but to give you a divisor. When someone says "we need 80,000 QPS," you immediately think "that's a small handful of boxes for simple reads, but if every request hits the database, that's the whole fleet's worth of database connections, so we'd better cache."

Memorize the latency ladder cold. Memorize the throughput table to within a factor of three. With those in your head, you never have to look anything up mid-review, and that is what makes estimation fast enough to actually do.

## Powers of two and the storage ladder

The second pillar of estimation is fluency with powers of two and the storage size ladder, because every storage and bandwidth estimate ends in bytes, and bytes climb by factors of about a thousand.

![A vertical stack of the storage ladder from kilobyte to petabyte, each rung labeled with its power of ten and a concrete example of data at that scale](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-2.webp)

Figure 2 shows that ladder. Here it is again in the form you should carry in your head, with the power-of-two and power-of-ten values side by side:

| Unit | Power of 10 | Power of 2 | Roughly |
| --- | --- | --- | --- |
| Kilobyte (KB) | 10³ | 2¹⁰ = 1,024 | A short text record, one URL row |
| Megabyte (MB) | 10⁶ | 2²⁰ | A photo, a small document, a song minute |
| Gigabyte (GB) | 10⁹ | 2³⁰ | A movie, fits easily in RAM |
| Terabyte (TB) | 10¹² | 2⁴⁰ | A large table-years; a big disk |
| Petabyte (PB) | 10¹⁵ | 2⁵⁰ | A full product's data corpus |
| Exabyte (EB) | 10¹⁸ | 2⁶⁰ | Hyperscaler-total territory |

The single most important habit here is to **treat 1,024 as 1,000**. Yes, a kibibyte is 1,024 bytes and a kilobyte is technically 1,000, and yes, the discrepancy compounds: by the time you reach a petabyte, 2⁵⁰ is about 1.13 × 10¹⁵, a 13% gap from 10¹⁵. For an envelope estimate, 13% is noise. Round 2¹⁰ to 10³ and move on. If your decision flips on a 13% difference, you are not doing an envelope calculation anymore — you are doing capacity planning, and you should be measuring real numbers, not estimating.

A second habit: **memorize a few "data costs" per item** so you can multiply quickly.

- A short text row (a user record, a URL mapping, a tweet's metadata): **~100 bytes to ~1 KB**.
- A chat message with metadata: **~100 bytes to ~1 KB** of stored bytes; ~200 bytes is a good default.
- A thumbnail: **~10–50 KB**. A compressed web photo: **~200 KB–2 MB**, call it **~1 MB** as a default. A raw phone photo: **~3–5 MB**.
- A minute of video at streaming quality: **~5–15 MB**; call it **~10 MB/min**, ~600 MB/hour.

These per-item costs are the multiplicands. When you know that a system stores 100 million photos at ~1 MB each, the storage is "100 million × 1 MB = 100 TB" in one mental step, because you have the per-item cost memorized and the unit ladder fluent. We will see this exact move repeatedly in the worked examples.

The third habit: **know the seconds in a day**. There are 86,400 seconds in a day. Round it to **~100,000 (10⁵)** for estimation. This single rounding makes the DAU-to-QPS conversion trivial: if every one of your N daily active users does one action per day, your *average* QPS is N / 10⁵. A hundred million DAU doing one action each is 10⁸ / 10⁵ = 10³ = 1,000 average QPS per action. Commit "86,400 ≈ 10⁵ seconds per day" to memory and the most common estimation step becomes a one-line division.

## The estimation workflow

Now we chain it together. The workflow always runs in the same order, and running it in the same order every time is what makes it reliable. You start from one number — usually daily active users or total users — and derive everything else by multiplication and division against your memorized constants.

![A pipeline showing the estimation flow from daily active users to queries per second to storage to bandwidth to servers, with the multiplier on each arrow](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-3.webp)

The five steps, with the question each one answers:

1. **DAU → QPS.** How many requests per second, on average *and at peak*? Multiply DAU by actions-per-user-per-day, divide by ~10⁵ seconds, then multiply by a peak factor (more on this next).
2. **QPS + per-item size → storage.** How many bytes per day, per year, and over the data's lifetime? Multiply the *write* QPS by bytes-per-write by seconds, then scale to days and years. Storage is driven by writes, not reads — reads don't add bytes.
3. **QPS + payload size → bandwidth.** How many bytes per second flow in and out? Multiply read QPS by response size for egress, write QPS by request size for ingress. This is where you spot CDN and egress-cost problems.
4. **QPS → servers.** How many machines? Divide *peak* QPS by per-server capacity (from the throughput table), then add headroom. Do this separately for the stateless tier and the datastore tier — they have very different per-box capacities.
5. **Working set → cache memory.** How much RAM to hold the hot set? Apply the 80/20 rule: roughly 20% of items get ~80% of reads, so caching the hot 20% (or even less) absorbs most read traffic. Size the cache to that hot set.

Two cross-cutting numbers govern this whole pipeline and deserve their own sections: the **peak factor** in step 1, and the **read-to-write ratio** that shapes steps 2 through 5. Get those two right and the rest is arithmetic. Get them wrong and your estimate is confidently off by an order of magnitude.

You do this in your head in a review, but it's worth encoding the workflow once so the chained multiplications are explicit and you can sanity-check your mental math. Here is the whole pipeline as a short, runnable function — not because you'd run it live, but because reading it cements the *order* of operations and the constants you fold in at each step:

```python
SECONDS_PER_DAY = 100_000          # 86,400 rounded for envelope math
KB, MB, GB, TB, PB = 1e3, 1e6, 1e9, 1e12, 1e15

def estimate(dau, actions_per_user_per_day, bytes_per_write,
             read_write_ratio, bytes_per_read, retain_years,
             peak_factor=5, server_qps=20_000, cache_hit_fraction=0.9):
    # Step 1: DAU -> QPS (writes and reads, average then peak)
    actions_per_day = dau * actions_per_user_per_day
    write_qps_avg = actions_per_day / SECONDS_PER_DAY
    read_qps_avg = write_qps_avg * read_write_ratio
    write_qps_peak = write_qps_avg * peak_factor
    read_qps_peak = read_qps_avg * peak_factor

    # Step 2: storage (writes only; reads add no bytes)
    bytes_per_day = write_qps_avg * SECONDS_PER_DAY * bytes_per_write
    storage_total = bytes_per_day * 365 * retain_years

    # Step 3: bandwidth (egress dominated by reads)
    egress_bps_peak = read_qps_peak * bytes_per_read       # bytes/sec at peak

    # Step 4: servers (size the read tier off PEAK, add headroom)
    read_servers = (read_qps_peak / server_qps) * 1.3      # +30% headroom

    # Step 5: cache misses that still hit the datastore
    db_read_qps_peak = read_qps_peak * (1 - cache_hit_fraction)

    return {
        "write_qps_peak": round(write_qps_peak),
        "read_qps_peak": round(read_qps_peak),
        "storage_TB": round(storage_total / TB, 1),
        "egress_Gbps_peak": round(egress_bps_peak * 8 / 1e9, 1),
        "read_servers": round(read_servers, 1),
        "db_read_qps_peak": round(db_read_qps_peak),
    }
```

The function is deliberately boring: it is just the five steps in order, with the constants (seconds per day, peak factor, headroom, hit rate) named so you can see exactly where each assumption enters. When you run the URL-shortener inputs through it below, the outputs match the by-hand arithmetic — which is the point. The value of writing it down is that the *named constants* are the assumptions, and naming them is what makes the estimate defensible.

## Peak versus average: the factor that breaks naive estimates

The most common estimation mistake — made by smart people constantly — is to compute the *average* QPS and then size the system for it. Traffic is not flat. If you provision for the daily average, you are under-provisioned by a large multiple exactly when it matters most: at the peak.

![A timeline of a daily traffic curve from a night trough at a fraction of average to a sharp evening peak at several times average](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-4.webp)

Real consumer traffic follows a daily curve like the one in figure 4. There is a deep trough in the small hours (maybe 0.3× average), a ramp through the morning, a midday plateau, and a sharp evening peak. For most consumer apps the **peak is roughly 2× to 10× the daily average**, with a common rule-of-thumb of **~5×** when you have nothing better to go on. The exact factor depends on the workload — a global app with users spread across time zones has a flatter curve and a lower peak factor; a single-country app where everyone opens it at 8 PM has a brutal one.

The reason this matters so much is that **you provision for the peak, but you pay for the peak too**. Figure 5 contrasts the two sizing approaches.

![A before-and-after comparison of average-based capacity sizing that melts under peak load versus peak-aware sizing with headroom](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-5.webp)

The discipline is simple and you should apply it every single time:

1. Compute **average QPS** first (it's the easy division: actions per day ÷ ~10⁵).
2. Multiply by your **peak factor** (default 5×, adjust for the workload) to get **peak QPS**.
3. Size servers and capacity for **peak QPS plus headroom** (typically +30% so you're not running at the redline; a system at 100% utilization has no room to absorb a retry storm or a node failure).
4. But size **storage and cost-over-time** off the **average** — because storage accumulates over the whole day, not just the peak.

That last point is a subtlety worth stating clearly: **peak drives compute and capacity; average drives storage and total bytes.** You don't store more bytes because of a peak — you store the day's total writes regardless of how they were distributed in time. So you use peak QPS to count servers and average QPS (or daily totals) to count gigabytes. Mixing those up is how people end up either with a melted fleet at 8 PM or a wildly over-estimated storage bill.

There is a real trade-off hiding in the headroom decision, and it is the central optimization tension of capacity planning: **provisioning headroom versus cost**. More headroom means you survive spikes and failures gracefully, but you pay for idle capacity 23 hours a day. Less headroom means a leaner bill but a system that tips over under a surge. Modern autoscaling softens this — you can scale the stateless tier with load and pay closer to the average — but autoscaling has a reaction time (often minutes), and a sharp spike can outrun it. Stateful tiers (databases, caches) scale much more slowly and usually have to be provisioned closer to peak. The senior move is to autoscale the cheap, stateless tier aggressively and provision the expensive, stateful tier for peak-plus-margin, accepting that the database is where you pay for headroom.

## Read-to-write ratio: the number that dictates the architecture

If peak factor is the number that breaks naive *capacity* estimates, the read-to-write ratio is the number that breaks naive *architecture*. Before you choose a single technology, you should know roughly how many reads you do per write, because that ratio decides where the bottleneck lives and therefore what you optimize.

Reads and writes stress completely different parts of a system, so the ratio between them tells you which part to harden:

- A **read-heavy** system (ratio like 100:1 or 1000:1 — a URL shortener, a news feed, most content sites) bottlenecks on serving reads. The levers are **caching, read replicas, and CDNs**. Writes are a rounding error; you can afford a slower, stronger write path because there are so few writes. This is the regime where a cache earns its keep dramatically — see [caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) for how to do it without getting burned.
- A **write-heavy** system (ratio near 1:1 or write-dominated — a chat system, a metrics ingestion pipeline, an IoT backend) bottlenecks on absorbing writes. The levers are **sharding the write path, write-optimized storage (LSM-trees), and batching**. A cache barely helps because the data is constantly changing; the question is how to spread writes across enough partitions. This is where [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) and [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) become the core of the design.

![A matrix mapping read-to-write ratios from read-heavy to write-dominated against example systems, their bottleneck, and the primary architectural lever](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-6.webp)

Figure 6 maps this out. The practical upshot for estimation: **as soon as you have the QPS, split it into read QPS and write QPS using the ratio, because they flow into different parts of the workflow.** Write QPS drives storage and ingest. Read QPS drives bandwidth, cache sizing, and the read-serving fleet. A senior never reports a single "QPS" number without immediately decomposing it, because the undifferentiated number hides the architectural decision.

One more reason the ratio dominates: it determines whether you can lean on **asynchronous replication and eventual consistency** (cheap, fast, available) or whether you need **synchronous, strongly consistent writes** (slower, more coordination). Read-heavy systems often tolerate slightly stale reads from replicas, which is enormously cheaper to scale. The consistency trade-offs are their own deep topic — [consistency models from linearizable to eventual](/blog/software-development/database/consistency-models-from-linearizable-to-eventual) and [CAP theorem and PACELC](/blog/software-development/database/cap-theorem-and-pacelc) cover the spectrum — but estimation is what tells you which regime you're even in.

## From DAU to QPS, carefully

The DAU-to-QPS conversion is the most error-prone step in the whole workflow, so it's worth slowing down on. The headline number you're usually handed is *total users* or *daily active users*, and neither is QPS. Getting from one to the other involves three multipliers that people routinely fumble.

**Multiplier one: active fraction.** "We have 500 million registered users" is not 500 million DAU. The daily active fraction of a registered base is often 10–40% for a healthy consumer product. If you're handed total users, knock it down to DAU first, or you'll over-estimate by 3–10×. Conversely, if you're handed DAU, don't inflate it to the registered base for load math — the inactive users don't generate requests today.

**Multiplier two: actions per active user per day.** This is the multiplier most worth interrogating, because it varies wildly by feature. A user might open the app 5 times a day but each open triggers 20 background API calls (feed refresh, presence, notifications, analytics beacons). So "actions" at the API layer is often 10–100× the user-visible interactions. A senior asks "actions per user per day *at which layer*?" — user-facing reads, or total API calls including the chatty background traffic? They differ by an order of magnitude, and the load you provision for is the API-layer number, not the user-intent number.

**Multiplier three: the peak factor**, which we covered — average to peak is 2–10×.

Stack the three and the spread is enormous: the same "500M users" can mean anywhere from a few thousand to a few million peak QPS depending on the active fraction, the actions-per-user, and the peak factor. This is exactly why you write the assumptions down. The headline number alone tells you almost nothing; the multipliers are the estimate.

#### Worked example: the same headline, two answers

Take "100M registered users" and run it two ways to see how much the multipliers matter.

*Conservative read:* 20% daily active = 20M DAU; 10 user-facing actions/day; ÷ 10⁵ s = 2,000 QPS average; ×5 peak = **~10,000 QPS peak.**

*Chatty read:* 40% daily active = 40M DAU; 50 API calls/day (background refresh, presence, telemetry); ÷ 10⁵ = 20,000 QPS average; ×5 peak = **~100,000 QPS peak.**

Same headline, a 10× spread in the answer — and 10× is the difference between "a handful of servers" and "a real fleet with a sharded backend." Neither read is wrong; they make different assumptions, and the assumptions are explicit, so the review can pick the right one. A junior reports one number with false confidence. A senior reports the range and the multiplier that drives it.

## Worked example: sizing a URL shortener

Time to run the full workflow on a real system. A URL shortener (think Bitly) is the canonical read-heavy example, so it's the perfect first case. The prompt: "Design a URL shortener handling 100 million new URLs per month."

#### Worked example: URL shortener capacity

**Step 0 — restate the inputs and assumptions.** This is the step people skip, and it's the one that makes the estimate defensible. I'll write my assumptions down so anyone in the review can challenge a specific one:

- 100M new URLs (writes) per month.
- Read-to-write ratio of **100:1** — short links exist to be clicked, and a popular link is clicked many times. (This is an assumption; if the reviewer thinks it's 10:1 or 1000:1, we change one number and rerun.)
- Each stored record (short code, long URL, owner, timestamps) is **~500 bytes**, round to ~0.5 KB.
- Retain data for **5 years**.
- Peak factor **5×**.

**Step 1 — writes per second.** 100M writes/month ÷ (30 days × ~10⁵ s/day) = 100M ÷ 3M ≈ **~33 writes/s average**. Round it: ~30–40 writes/s. At 5× peak, ~**170 writes/s peak**. That is *tiny*. A single database handles this in its sleep.

**Step 2 — reads per second.** At 100:1, average reads = 33 × 100 = **~3,300 reads/s**. Peak reads at 5× = **~16,500 reads/s**, call it ~**17k QPS peak**. Now we have the real workload: a trickle of writes, a meaningful but modest stream of reads.

**Step 3 — storage.** Writes per year = 100M/month × 12 = **1.2 billion URLs/year**. Over 5 years = **6 billion records**. At 0.5 KB each: 6 × 10⁹ × 500 bytes = 3 × 10¹² bytes = **~3 TB** over five years. That is the entire dataset. Three terabytes fits on a single modern SSD with room to spare. **We do not need to shard for storage.** That negative result is the most important sentence in the whole estimate — it kills any premature "let's use a distributed datastore" proposal on the spot.

**Step 4 — bandwidth.** A redirect response is small — an HTTP 301 with a Location header, ~500 bytes including headers. Egress = 17k reads/s × 500 bytes = ~8.5 MB/s ≈ **68 Mbps** at peak. That's a fraction of a single 10 Gbps NIC. Ingress (writes) is negligible. Bandwidth is a non-issue here.

**Step 5 — servers.** Redirects are about as cheap as a request gets — look up a key, return a header. A single server can comfortably do tens of thousands of these per second if the lookup hits a cache. At 17k peak read QPS, **two app servers** (one for capacity, one for redundancy) plus headroom is plenty. Call it 2–3 stateless boxes behind a load balancer (see [load balancing from L4 to L7](/blog/software-development/system-design/load-balancing-from-l4-to-l7) for how to spread that traffic).

**Step 6 — cache.** Here's where the read-heavy nature pays off. By the 80/20 rule, a small fraction of links carry most of the clicks. Suppose we want the cache to absorb the hot set — say the most-accessed 20% of the URLs accessed in a typical day. Even better, we can cache by *recency and popularity*: the working set of "links being actively clicked right now" is far smaller than the full 6 billion records. If we cache, say, the hot 100 million entries at ~0.5 KB each (key + value), that's 100M × 0.5 KB = **~50 GB** of cache — comfortably one Redis instance, or a small cluster with headroom. A 90%+ cache hit rate on a read-heavy system this skewed is realistic, which means the database sees only ~1,700 read QPS even at peak. The single database handles writes (~170/s) and cold reads (~1,700/s) easily.

**The verdict.** A URL shortener at this scale is a **single-database, cache-fronted, two-or-three-server system** for the first several years. No sharding, no distributed datastore, no multi-region complexity. The estimate took five minutes and prevented a six-month over-engineering detour. *That* is the payoff. The dataset will need attention around year 5+ as it approaches the limits of one box, and we'll revisit then — estimation also tells you *when* to revisit.

Dropping the same inputs into the estimate function from earlier confirms the by-hand math — the headline write rate here is the *monthly* figure expressed per second, so the per-second action rate is small:

```python
# 100M writes/month = 100M / 30 days = ~3.3M writes/day across the user base.
# Express as DAU x actions/day however you like; the product is what matters.
print(estimate(
    dau=3_300_000,            # treat as "3.3M write-actions/day"
    actions_per_user_per_day=1,
    bytes_per_write=500,
    read_write_ratio=100,
    bytes_per_read=500,
    retain_years=5,
    peak_factor=5,
    cache_hit_fraction=0.9,
))
# -> {'write_qps_peak': 165, 'read_qps_peak': 16500, 'storage_TB': 3.0,
#     'egress_Gbps_peak': 0.07, 'read_servers': 1.1, 'db_read_qps_peak': 1650}
```

Every output lines up with the hand arithmetic: ~165 write QPS peak, ~16.5k read QPS peak, ~3 TB over five years, ~0.07 Gbps egress, roughly one read server before redundancy, and ~1,650 QPS reaching the database after a 90% cache. The code didn't *change* the answer — it just made every assumption a named, auditable input. That is the entire discipline in one function.

#### Worked example: URL shortener rough monthly bill

Now the cost dimension, because "it fits on one box" still costs money and a senior quotes a number. I'll use round public-cloud rates (your provider and committed-use discounts will shift these, but the proportions hold):

- **Compute:** 3 app servers + 1 database server + 1 cache server = 5 instances. At ~\$50–\$150/instance/month for modest boxes, call it ~\$500/month for compute. Round generously: **~\$500–\$700/mo**.
- **Storage:** 3 TB on block/SSD storage at ~\$0.10/GB/month = 3,000 GB × \$0.10 = **~\$300/mo**, growing as data accumulates.
- **Egress:** Peak 68 Mbps doesn't run at peak all day; average egress might be ~20 Mbps. Over a month that's roughly 20 Mbps × 2.6M seconds ÷ 8 ≈ ~6.5 TB/month of egress. At ~\$0.08/GB, that's 6,500 GB × \$0.08 = **~\$520/mo**. Notice this rivals the compute cost — and we have a *tiny* response size. For a URL shortener egress is manageable; keep this number in mind for the photo example, where it explodes.
- **Total:** roughly **\$1,300–\$1,500/month** all-in. A two-pizza side project's worth of infrastructure serves 100M URLs/month. Knowing that up front sets a sane budget and immediately flags that egress and storage together already outweigh compute.

## Worked example: sizing a photo-sharing feed

Now a heavier system, to show how the same workflow scales up and where the bottleneck *moves*. The prompt: "Design the storage and delivery for a photo-sharing app with 100 million daily active users."

This one is also read-heavy, but the per-item payload is a million times bigger than a URL row, so bandwidth and storage — not QPS — become the story. And it introduces the CDN, which is the single most important cost optimization for media-heavy systems.

![A graph showing a photo feed where uploaders write to origin storage, a CDN edge absorbs most viewer reads, and only cache misses pull from origin](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-7.webp)

#### Worked example: photo feed storage and bandwidth

**Step 0 — assumptions:**

- 100M DAU.
- Each user **uploads ~2 photos/day** on average and **views ~200 photos/day** (scrolling a feed). That's a ~100:1 read-to-write ratio at the photo level — read-heavy, as expected.
- A stored photo, after server-side compression and generating a few thumbnail sizes, costs **~1.5 MB total** across all variants. Round to **~1.5 MB/photo** stored.
- A photo *delivered* to a feed is the web-optimized variant, **~300 KB** average (most feed views are not full-resolution).
- Retain photos **forever** (this is the killer for storage — more below).
- Peak factor **5×**.

**Step 1 — write (upload) QPS.** 100M users × 2 uploads/day = **200M uploads/day**. ÷ 10⁵ s = **~2,000 uploads/s average**, ~**10,000 uploads/s peak**. Each upload is one object write — modest QPS, but each carries ~1.5 MB.

**Step 2 — read (view) QPS.** 100M × 200 views/day = **20 billion views/day**. ÷ 10⁵ = **~200,000 views/s average**, **~1,000,000 views/s peak**. A million reads per second is a lot — but, crucially, photos are *immutable* once uploaded, which makes them perfectly cacheable. This is the whole game.

**Step 3 — storage growth.** 200M uploads/day × 1.5 MB = 300M MB/day = **~300 TB/day**. Per year: 300 TB × 365 ≈ **~110 PB/year**. *Per year.* This is the number that should make you sit up. The compute is trivial relative to this. Photo storage is an unbounded, ever-growing liability, and "retain forever" means the storage bill grows linearly and permanently. The senior questions to ask the room immediately: do we really retain at full fidelity forever? Can cold photos move to cheaper archival storage (glacier-class) after 90 days? Can we tier aggressively? The estimate surfaces the real cost driver before a single byte is stored.

**Step 4 — bandwidth, and why the CDN is non-negotiable.** Egress at peak = 1,000,000 views/s × 300 KB = **300,000 MB/s = 300 GB/s = 2,400 Gbps**. That is 240 × 10-Gbps NICs *of egress at peak*, served continuously. If that traffic came off your origin servers, you would need a colossal origin fleet and an even more colossal egress bill. This is why media systems live and die by the CDN. As figure 7 shows, the CDN serves the immutable photos from edge caches close to users; only cache *misses* (cold or newly uploaded photos) pull from origin. At a 95% edge hit rate, the origin serves only 5% of those reads = ~50,000 reads/s pulling from origin, and 95% of the 2,400 Gbps egress is served from the CDN's network, not yours. The CDN turns an impossible origin-bandwidth problem into a tractable one — at the cost of CDN egress fees, which we'll price next.

**Step 5 — servers.** Uploads at 10k/s peak, each doing some compression and thumbnail generation, is CPU work — call it a few hundred upload-handling servers depending on how heavy the image processing is. Feed-metadata serving (the JSON that says "here are the photo IDs in your feed") is separate and far lighter than the photo bytes themselves; the heavy lifting is the bytes, and the CDN handles those. The fan-out of building each user's feed is its own design problem — push vs pull, covered when we discuss feeds — but the *byte delivery* is dominated by the CDN.

#### Worked example: photo feed monthly bill (egress is the silent killer)

Now the cost, where this example earns its place. Compute is the *small* number here:

- **Storage:** We're adding ~110 PB/year and keeping it. After year one, ~110 PB at object-storage rates of ~\$0.02/GB/month = 110 × 10⁶ GB × \$0.02 = **~\$2.2M/month**, and it grows by ~\$2.2M/month *every year* if we retain forever. Tiering cold data to archival storage (~\$0.004/GB/month, 5× cheaper) on the 90% of photos that are rarely viewed after their first week could cut this dramatically — easily saving a million-plus dollars a month at this scale. This is exactly the kind of optimization the estimate makes visible.
- **CDN egress:** The 2,400 Gbps peak averages out over a day; say sustained average egress is ~800 Gbps = 100 GB/s. Over a month: 100 GB/s × 2.6M s = **~260,000,000 GB ≈ 260 PB/month** of CDN egress. At a negotiated CDN rate of even \$0.01–\$0.02/GB (large customers negotiate hard down from list prices of \$0.05–\$0.08), that's 260M GB × \$0.015 = **~\$3.9M/month** in egress. **Egress is the single largest line item, and it's invisible until the invoice arrives** — hence "the silent budget killer." This number alone justifies a full-time team negotiating CDN contracts and building multi-CDN strategies, which is exactly what photo and video companies do.
- **Compute:** A few hundred upload servers plus feed-serving boxes is, by comparison, *noise* — perhaps \$100k–\$300k/month. The thing engineers instinctively count (servers) is the smallest cost; the things they forget (storage growth and egress) are the budget.
- **Total:** order of **\$6–8M/month** and rising, dominated by storage and egress. The architecture decisions that move that number are tiering policy and CDN economics — *not* which web framework serves the feed.

The lesson from this example is the one that separates senior estimates from junior ones: **for media systems, count the bytes and the egress first, the QPS and servers second.** The bottleneck and the cost are both in the bytes.

## Worked example: sizing a chat system

The third example flips the read-to-write ratio toward writes and introduces fan-out, which is where chat and feed systems get genuinely hard. The prompt: "Size the storage and write load for a chat system with 500 million daily active users."

#### Worked example: chat messages and write load

**Step 0 — assumptions:**

- 500M DAU.
- Each user **sends ~40 messages/day** on average. (Heavy texters send hundreds; lurkers send zero; 40 is a reasonable mean for a large messaging platform.)
- A stored message with metadata (sender, recipient/conversation, timestamp, body) averages **~200 bytes** — most chat messages are short.
- Messages are mostly 1:1 and small-group, but we'll account for **fan-out**: a message to a group of size G must be delivered to G recipients.
- Retain messages **forever** (people scroll back years).
- Peak factor **5×**.

**Step 1 — message write rate.** 500M × 40 = **20 billion messages/day**. ÷ 10⁵ = **~200,000 messages/s average**, **~1,000,000 messages/s peak**. This is a genuinely write-heavy system: a million message writes per second at peak, sustained. Unlike the URL shortener, a single database cannot absorb this. We are in shard-everything territory from day one. The relevant mechanism deep-dives are [database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) and [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) for spreading those writes; the architect's view of doing it live is [partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime).

**Step 2 — storage growth.** 20 billion messages/day × 200 bytes = 4 × 10¹² bytes/day = **~4 TB/day** of message bodies. Per year: 4 TB × 365 = **~1.5 PB/year**, retained forever. Smaller than the photo example (text is cheap), but it grows without bound, and at 1.5 PB/year accumulating, you will be sharded across many nodes within the first year. The storage-engine choice matters: a write-heavy, append-mostly workload like a message log is a textbook fit for LSM-tree storage ([LSM-trees as write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines)), which is exactly why systems like this often sit on Cassandra or ScyllaDB rather than a B-tree-based SQL database.

**Step 3 — the fan-out multiplier.** Here is where chat differs from the previous examples and where naive estimates go badly wrong. The 1M messages/s is the *write* (send) rate. But each message must be *delivered* to recipients, and delivery is a separate, larger number. For 1:1 chat, fan-out is 1× (one recipient). For a group of 100, one send becomes 100 deliveries. If even a modest fraction of traffic is group messages, the delivery rate dwarfs the send rate.

Suppose 10% of messages go to groups averaging 50 members. Then deliveries = (90% × 1 recipient) + (10% × 50 recipients) per send = 0.9 + 5 = ~5.9 deliveries per message on average. So delivery QPS = 1M sends/s × 5.9 ≈ **~6M deliveries/s at peak**. *That* is the number that sizes your delivery/fan-out tier, your connection servers, and your push infrastructure — and it's 6× the send rate. A senior reports both numbers and is explicit about the fan-out multiplier, because it's the difference between provisioning for 1M/s and 6M/s. The deeper this goes — push fan-out on write vs pull on read, the "celebrity problem" where one sender has millions of recipients — is its own design, but estimation is what reveals that fan-out, not raw sends, is the dominant load.

**Step 3b — the tail of the fan-out distribution.** The average fan-out (5.9×) sizes the steady-state delivery tier, but it dangerously hides the *tail*, and the tail is what causes outages. Suppose the platform has broadcast groups or channels with 100,000 members. A single message to one of those is 100,000 deliveries from one write — a 100,000× fan-out spike concentrated on whatever shard owns that channel. If a few such broadcasts land in the same second, you get a localized burst of millions of deliveries against a single partition while the rest of the fleet sits idle. This is the "hot key" problem, and it's invisible in the averaged estimate. The senior move is to estimate the worst-case fan-out *separately*: max group size × messages those groups send per second. If that number is large, the design needs a different delivery path for large groups (fan-out-on-read, or a pub/sub broadcast tier) than for 1:1 chat. You cannot discover that from the 5.9× average; you have to deliberately look at the tail. **What breaks at the tail is never the average — it's the worst case you forgot to estimate.**

**Step 4 — connections and memory.** A chat system holds **persistent connections** (WebSocket/long-poll) for every online user so it can push messages instantly. With 500M DAU and, say, 50M concurrently connected at peak, and each connection costing ~10 KB of server memory (buffers, state), that's 50M × 10 KB = **~500 GB of connection state**, spread across the connection-server fleet. If one connection server holds ~100k connections (a reasonable target with efficient async I/O), you need ~500 connection servers just to hold the sockets at peak. That fleet-sizing falls straight out of the estimate. Note the second-order cost: those 500 servers are *stateful* (they hold the sockets), so they can't autoscale away during the trough the way a stateless tier can — you pay for a good fraction of them around the clock. The connection fleet is one of the places a chat system pays for its always-on nature.

**The verdict.** A chat system is **write-heavy and fan-out-dominated**: shard the message store from day one, choose write-optimized storage, size the *delivery* tier off the fan-out-multiplied rate (~6M/s) rather than the send rate (~1M/s), and provision a large connection-server fleet for persistent sockets. The read-to-write ratio and the fan-out multiplier — not the headline DAU — are what determine the shape. Contrast this with the URL shortener, where the same workflow concluded "one database, don't shard." Same five steps, opposite architecture, because the ratios were different. **That contrast is the entire point of estimation.**

## How estimation drives the architecture

Step back and notice what the three examples actually produced. The same workflow, run on three systems, produced three completely different architectures — and the deciding factor was always a *number*, surfaced by the estimate, not a technology preference.

![A decision tree asking whether the working set fits in one box's RAM or disk, branching to single-box, replicated, or sharded architectures](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-8.webp)

Figure 8 captures the single most consequential fork estimation resolves: **does it fit on one box?** This is the question that decides whether you live in the (vastly simpler) single-node world or the (vastly more complex) distributed world.

- If the **working set fits in RAM** (tens of GB), you can serve it from memory on a single machine. The URL shortener's hot cache (~50 GB) lives here.
- If the **full dataset fits on one machine's disk** (single-digit TB) but not RAM, one box with SSD plus a read replica or two for availability still does the job. The URL shortener's full 3 TB dataset lives here.
- If the dataset or write rate **exceeds one machine** (tens of TB and up, or write rates past what one node absorbs), you must shard. The photo system (110 PB/year) and chat system (1M writes/s) are unambiguously here.

The reason this fork dominates is that crossing it changes *everything*: sharding introduces partition keys, cross-shard queries, rebalancing, distributed transactions, and a whole category of consistency problems that simply don't exist on one box. A senior fights to stay on the single-node side of this line as long as the numbers allow, because every problem is easier there. Estimation is the tool that tells you, honestly, which side of the line you're on — and, just as importantly, *when* a growing system will cross it, so you can plan the migration before it's an emergency.

The same logic applies to every other architectural decision:

- **Cache or not?** Driven by read QPS and the read-to-write ratio. High read ratio + skewed access (80/20) → cache pays off enormously (URL shortener). Write-heavy + uniformly changing data → cache barely helps (chat).
- **CDN or not?** Driven by egress bandwidth of large, cacheable, immutable objects. Big immutable bytes served to many users → CDN is non-negotiable (photos). Tiny dynamic responses → CDN optional.
- **Replicas or shards?** Driven by whether you're read-bound (add replicas — see [replication strategies and their failure modes](/blog/software-development/system-design/replication-strategies-and-their-failure-modes)) or write/size-bound (add shards). The read-to-write ratio decides.
- **One region or many?** Driven by latency budgets (that 150 ms cross-region number) and availability requirements, weighed against the cost and consistency complexity of multi-region.

In every case, you don't argue the decision in the abstract. You estimate the number, and the number argues the decision for you. That is what it means for estimation to "drive the architecture."

## Sanity-checking against physical limits

A separate but related use of these numbers is **sanity-checking a proposed design against physics**. Sometimes a design is internally consistent but quietly impossible, and the envelope numbers catch it.

A few physical-limit checks worth running on any design:

- **Can a single NIC carry the traffic you're routing through one box?** A 10 Gbps NIC is ~1.25 GB/s. If your design funnels 5 GB/s of egress through a single node, the design is broken regardless of how elegant the code is. You need either multiple NICs, multiple nodes, or a CDN.
- **Can a single disk sustain the write throughput?** One SSD does maybe 500 MB/s–2 GB/s sequential, far less random. If your write rate × payload exceeds that, one disk can't keep up and you need striping or sharding.
- **Does the latency budget physically allow the call pattern?** If your p99 budget is 100 ms and your design makes three *sequential* cross-region calls at 150 ms each, you've already spent 450 ms before doing any work. The design is physically impossible at that latency target; you must parallelize, cache, or move data closer.
- **Does the connection count exceed what a server can hold?** File descriptors, memory per connection, and ephemeral port ranges all cap how many connections one box sustains. If your design implies 2 million connections on one server, check that against ~100k–1M practical limits per box.

These checks take seconds and they catch the embarrassing class of design that looks fine on the whiteboard but violates a hardware limit. The latency ladder and throughput table are precisely the constants you check against. This is also where you stress-test: **what breaks at 10×?** Run your estimate again with every input multiplied by ten and see which physical limit you hit first. For the URL shortener at 10×, you're at ~170k read QPS and 30 TB — still one box on storage, but now you need a real read-serving fleet and the cache becomes load-bearing. For the photo system at 10×, you're past a *petabyte per day* and the egress bill alone could fund a small company; the design doesn't break so much as the *economics* do, which is its own kind of breakage. Running the 10× check tells you where each system's true ceiling is.

## Trade-offs: precision versus speed, and headroom versus cost

Estimation itself involves trade-offs, and a senior is deliberate about them. The two that matter most are summarized below, then unpacked.

| Decision | What you gain | What you pay | When it wins |
| --- | --- | --- | --- |
| Round aggressively (factor-of-2 estimates) | Speed; can do it live in a review; covers more scenarios | Imprecision; can be off by 2–3× | Early architecture, go/no-go decisions, the fits-in-RAM question |
| Estimate precisely (real measurements, careful units) | Accuracy; defensible budget numbers | Time; needs data you may not have yet | Final capacity planning, cost commitments, contract negotiation |
| High provisioning headroom | Survives spikes, failures, retry storms gracefully | Idle capacity paid for ~23 h/day | Stateful tiers, hard SLOs, unpredictable traffic |
| Lean provisioning + autoscaling | Lower bill, pay near average | Reaction lag; sharp spikes can outrun scaling | Stateless tiers, predictable or slow-moving load |

**Precision versus speed.** The whole value proposition of back-of-the-envelope estimation is that it's *fast*, and the price of speed is precision. You round 86,400 to 100,000, you round 1,024 to 1,000, you assume a 5× peak factor without data. The result is good to within a factor of 2–3 — which is exactly the precision the big decisions need and no more. The senior skill is knowing **when factor-of-2 is enough and when it isn't.** Choosing whether to shard? Factor-of-2 is plenty — the answer is "obviously yes" or "obviously no" or "we'll revisit in a year," and none of those flip on 30%. Committing to a three-year reserved-instance cloud contract worth millions? Now you measure real traffic, because a 30% error is real money and the decision is hard to reverse. The mistake in both directions is real: over-precision wastes time and creates false confidence in shaky inputs; under-precision on an irreversible commitment burns budget. Match the rigor to the stakes and the reversibility.

**Headroom versus cost.** The other live trade-off, covered earlier: more headroom buys resilience and costs idle capacity; less headroom saves money and risks tipping over. The senior resolution is *tier-specific*: autoscale the cheap stateless tier aggressively and run it lean, accepting some reaction lag; provision the expensive stateful tier (databases, caches, stateful brokers) for peak-plus-margin, because it scales slowly and is where an under-provision causes a real outage. You don't pick one global headroom number — you pick the right one per tier based on how fast that tier can scale and how catastrophic running out is.

There's a meta-point here that defines the senior posture: **an estimate is a claim with stated assumptions, not an oracle.** Every number above rests on an assumption — peak factor, read ratio, per-item size, retention. The discipline isn't hiding those assumptions behind a confident number; it's writing them down so the review can attack the *assumption* rather than the conclusion. When a reviewer says "I think it's more like 200 messages a day, not 40," you change one input and rerun in ten seconds. That's the difference between an estimate (defensible, adjustable) and a guess (a single number you have to defend to the death). This connects directly to [turning vague asks into requirements and SLOs](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos): the requirements give you the inputs, and the estimate turns them into a design. And once you have numbers, naming the resulting trade-offs precisely is its own skill, covered in [articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond).

## The cost dimension in depth

Compute, storage, and egress are the three legs of almost every cloud bill, and estimation should always end with a rough split across them, because *where* the cost concentrates tells you *what* to optimize.

![A matrix of cloud cost components showing rough rates, what each scales with, and which line items quietly dominate the bill](/imgs/blogs/back-of-the-envelope-estimation-for-system-design-9.webp)

Figure 9 lays out the cost components and, crucially, what each one *scales with* and which ones are dangerous. The senior mental model:

- **Compute** is the cost engineers instinctively reach for, and it's usually *not* the dominant line item at scale. It scales with peak QPS and CPU work per request. It's predictable, autoscalable, and increasingly cheap. Pricing it is easy: peak servers × per-server rate.
- **Storage** is the slow killer. It scales with *the integral of your write rate over time* — it accumulates and, if you retain forever, never goes down. The photo example showed storage at \$2.2M/month *and rising \$2.2M/month every year*. The optimization levers are retention policy (delete or don't store what you don't need), tiering (move cold data to archival storage at ~5× lower cost), and compression. A senior asks "how long do we really keep this?" early, because the answer multiplies a per-year cost by the number of years.
- **Egress** is the silent killer — the line item engineers forget entirely and that frequently dominates media-heavy bills. It scales with *bytes served to the outside world*: read QPS × response size. Within a cloud region, inter-AZ transfer is cheap-ish but not free (~\$0.01/GB) and adds up with replication. To the internet, egress is expensive (~\$0.05–\$0.09/GB list, negotiable for big customers). The optimizations are CDN offload (serve from edge, not origin), compression (fewer bytes), caching (don't re-fetch), and keeping traffic in-region. The photo example's \$3.9M/month egress is the canonical cautionary tale: you can build a beautiful system and have its bill dominated by a number nobody costed.

The senior habit is to **end every estimate with a one-line cost split** — "roughly X on compute, Y on storage, Z on egress" — because that split immediately points at the optimization with the highest leverage. For a URL shortener, all three are small and roughly balanced; you optimize nothing and ship. For a photo system, storage and egress are 95% of the bill; every optimization dollar goes there, and arguing about the web framework is a waste of breath. The estimate doesn't just size the system — it tells you *where to spend your engineering attention*.

## Validating an estimate against reality

An estimate made on a whiteboard is a hypothesis. Once the system runs — or once you have a comparable system to measure — you should close the loop and check whether reality landed inside your factor-of-2 band. This is what keeps your constants honest: if your guessed peak factor was 5× but production shows 8×, you update the default you carry into the next estimate. The most valuable engineers have *calibrated* constants, tuned by years of comparing estimates to measurements.

For a running system, the measurements you want map one-to-one onto the workflow steps. Peak QPS comes straight off the load balancer or metrics system; storage growth comes from the database; egress comes from the cloud billing or the CDN dashboard. A few one-liners get you the real numbers to compare against your estimate:

```bash
# Real peak QPS over the last week, from access logs (1-minute buckets -> per-second).
# Pull the busiest minute and divide by 60 to get peak QPS.
awk '{print substr($4, 2, 17)}' access.log \
  | sort | uniq -c | sort -rn | head -1 \
  | awk '{printf "peak: %.0f req/s (%d in that minute)\n", $1/60, $1}'

# Real average -> peak factor: peak QPS / mean QPS over the same window.
# If this prints > your assumed peak_factor, your sizing was too lean.
```

And the storage-growth rate — the number that drives the scariest line of the cost estimate — comes straight from the database:

```sql
-- Daily write volume: are we accumulating bytes at the rate we estimated?
-- Compare bytes_per_day here against (write_qps_avg * 86400 * bytes_per_write).
SELECT
  date_trunc('day', created_at)        AS day,
  count(*)                             AS rows_written,
  pg_size_pretty(sum(octet_length(payload))) AS bytes_written
FROM messages
WHERE created_at >= now() - interval '14 days'
GROUP BY 1
ORDER BY 1;
```

If the measured numbers sit within ~2× of the estimate, the estimate did its job — it got you to the right architecture and the right order-of-magnitude budget, which is all an envelope calculation promises. If they're off by more than ~3×, don't shrug: find out *which assumption* was wrong (active fraction? actions per user? payload size? peak factor?) and fix that constant going forward. The discipline isn't being right the first time; it's being calibrated over time. An estimate you never validate is just a guess with extra steps.

## Case studies

Real systems and incidents make the estimates concrete. Each of these illustrates an estimation lesson that teams learned, sometimes the hard way.

### Discord and the scale of message storage

Discord has publicly written about storing trillions of messages. Their journey is a live demonstration of the chat-system estimate: a write-heavy, ever-growing message store eventually forces a move to write-optimized, horizontally-scalable storage. They famously migrated their message store from MongoDB to Cassandra and later to ScyllaDB as the message volume grew, precisely because a write-heavy workload at that scale needs LSM-tree-based, shard-from-the-start storage — exactly what our chat estimate predicted ("shard from day one, choose write-optimized storage"). The lesson: when your envelope math says "millions of writes per second, growing forever," it is telling you to pick the storage engine and partitioning strategy *first*, because retrofitting them under load is among the hardest migrations there is.

### Netflix, Open Connect, and egress economics

Netflix delivers an enormous share of internet video traffic, and the defining decision of their infrastructure was to build their own CDN, Open Connect, placing caching appliances inside ISP networks. The reason is pure egress economics: at their scale, paying commercial CDN or transit rates for petabytes of video egress would be ruinous, and serving from boxes physically inside the ISP both cuts cost and improves quality. This is the photo-feed lesson at civilization scale — for a media company, **egress is the business problem**, and the architecture (a custom CDN embedded in last-mile networks) is shaped entirely by the bandwidth and cost estimate. If your envelope math shows egress dominating the bill, expect to invest heavily in offload, exactly as Netflix did.

### The over-provisioning and under-provisioning failure modes

Two opposite war stories that recur across the industry. On the under-provisioning side: teams that sized for *average* load and got destroyed by a predictable peak — the classic retail site that runs fine for eleven months and falls over on the biggest shopping day of the year, because nobody multiplied by the peak factor. The cap on capacity wasn't discovered until the traffic curve found it. On the over-provisioning side: teams that *over-estimated*, sharded a database that one node would have handled for years, and spent the next eighteen months fighting cross-shard queries and rebalancing complexity for scale they didn't have. Both failures come from the same root cause: not doing the estimate, or doing it without the peak factor and the fits-in-one-box check. The estimate is cheap insurance against both expensive mistakes.

### Twitter's timeline fan-out

Twitter's timeline architecture is the canonical fan-out case study, and it maps directly onto the chat example's fan-out multiplier. For most users, Twitter fans out a tweet on *write* — pushing it into each follower's materialized timeline — which is great for the average user but catastrophic for celebrities with tens of millions of followers, where one tweet means tens of millions of writes. Their solution was a *hybrid*: fan-out-on-write for normal accounts, fan-out-on-read (merge the celebrity's tweets in at read time) for the high-follower accounts. The estimation lesson is that **the fan-out multiplier is not uniform** — averaging it hides the tail, and the tail (the celebrity, the giant group chat) is where the system breaks. When you estimate fan-out, estimate the *worst case* fan-out separately from the average, because the worst case is what dictates the architecture.

## When to reach for this (and when not to)

Back-of-the-envelope estimation is the right tool in a specific set of situations, and using it outside that set wastes effort or creates false confidence.

**Reach for it when:**

- You're in a **design review or interview** and need to convert a vague scale into numbers that the design can be tested against. This is the primary use; do it every time.
- You're making an **early, hard-to-reverse architecture decision** — shard or not, cache or not, CDN or not, one region or many. The estimate's job is to tell you which side of the line you're on.
- You need a **rough budget** before provisioning, to set expectations and spot where cost will concentrate.
- You want a **sanity check** on someone else's design — does it violate a physical limit? Run the latency and throughput checks.
- You're deciding **what to optimize**. The cost split tells you where the leverage is.

**Don't lean on it (use measurement instead) when:**

- You're **committing real money to an irreversible contract** (multi-year reserved capacity). Now the factor-of-2 imprecision is a budget risk; measure real traffic.
- You're **tuning a hot path for a specific p99 target**. Estimation gets you the order of magnitude; hitting a precise latency SLO requires profiling and load testing the actual system, not a whiteboard.
- The **workload is genuinely novel** and you have no basis for the per-item costs and ratios — your assumptions would be fiction. Build a prototype, measure, *then* estimate forward from real numbers.
- You're **past the design phase and into capacity planning** for a running system. You have real metrics now; use them. Estimation is for when you don't yet have data, not a substitute for the data you do have.

The unifying principle: **estimation is for decisions, measurement is for commitments and tuning.** When the decision is reversible and the stakes are bounded, estimate and move fast. When the stakes are high and irreversible, the estimate gets you in the right neighborhood and then you measure to nail it down. Knowing which situation you're in is itself a senior judgment.

## The estimation cheat sheet

Pulling the whole toolkit into one place you can drill until it's reflexive. These are the constants and the moves; everything in this post is an application of them.

**Constants to memorize cold:**

| Quantity | Value to use | Why |
| --- | --- | --- |
| Seconds per day | ~10⁵ (86,400) | The DAU → QPS divisor |
| L1 / RAM / SSD / disk-seek | ~1 ns / 100 ns / 16 µs / 10 ms | The latency cliffs |
| Datacenter / cross-region RTT | ~0.5 ms / ~150 ms | In-region cheap, cross-region expensive |
| One commodity box, simple req | ~10k–50k QPS | The server divisor |
| One SSD sequential | ~500 MB/s–2 GB/s | The disk-throughput check |
| One 10 GbE NIC | ~1.25 GB/s | The bandwidth-per-box check |
| Storage ladder step | ~1000× per rung (KB→MB→GB→TB→PB) | Powers of two ≈ powers of ten³ |
| Peak factor | ~5× average (range 2–10×) | Size compute for this |
| Egress cost (to internet) | ~\$0.05–\$0.09/GB list | The silent budget killer |
| Object storage | ~\$0.02/GB/month | Multiply by years retained |

**The moves, in order:**

1. Knock total users down to **DAU** (active fraction 10–40%).
2. Multiply by **actions/user/day** *at the API layer* (often 10–100× user-visible).
3. Divide by **10⁵** → average QPS. Multiply by **peak factor** → peak QPS.
4. Split QPS into **read and write** via the ratio. Reads → bandwidth, cache, read fleet. Writes → storage, ingest.
5. Storage = write QPS × bytes/write × seconds × days × **years retained**.
6. Bandwidth (egress) = read QPS × response bytes. Check it against one NIC; if it's huge and the bytes are cacheable, **CDN**.
7. Servers = **peak** QPS ÷ per-box capacity × 1.3 headroom. Separate the stateless tier from the datastore tier.
8. Cache = hot-set size (80/20 rule); a high hit rate shrinks the DB read load by ~10×.
9. **Does it fit on one box?** RAM (tens of GB) → in-memory; disk (single-digit TB) → one box + replicas; bigger → shard.
10. End with the **cost split**: compute / storage / egress. The biggest leg is where you optimize.

If you can run those ten moves from memory against any one-line prompt, you can size a system in a review in under five minutes and defend every number in it. That fluency — not knowing the fashionable database — is what reads as senior.

## Key takeaways

- **Memorize the latency ladder and a throughput table.** L1 ~1 ns, RAM ~100 ns, SSD ~16 µs, datacenter round trip ~0.5 ms, disk seek ~10 ms, cross-region ~150 ms. One commodity box does tens of thousands of QPS for simple requests; storage units climb ~1000× per rung. These constants make estimation fast enough to do live.
- **Round aggressively: 86,400 ≈ 10⁵ seconds/day, 1,024 ≈ 1,000.** Envelope estimates are good to a factor of 2–3, which is exactly the precision that order-of-magnitude decisions need. Precision beyond that is wasted effort unless you're committing money.
- **Always run the same workflow: DAU → QPS → storage → bandwidth → servers → cache.** Same five steps every time, starting from one number and chaining multiplications against memorized constants.
- **Peak is 2–10× average (default ~5×); size compute for peak, storage for average.** Sizing for the average is the most common estimation error — it under-provisions you by a large multiple exactly at the peak.
- **The read-to-write ratio dictates the architecture before any technology choice.** Read-heavy → cache, replicas, CDN. Write-heavy → shard, LSM storage, batch. Always decompose QPS into read and write QPS.
- **The decisive question is "does it fit on one box?"** Stay on the single-node side of that line as long as the numbers allow — sharding introduces a whole universe of complexity. Estimation tells you which side you're on and when you'll cross.
- **For media systems, count the bytes and egress first.** Storage growth and egress, not compute, dominate the bill — and egress is the silent killer that's invisible until the invoice. End every estimate with a compute/storage/egress cost split to find the optimization with the highest leverage.
- **Estimate the worst-case fan-out separately from the average.** The celebrity and the giant group chat — the tail of the fan-out distribution — is what breaks the system, and averaging hides it.
- **An estimate is a claim with stated assumptions, not an oracle.** Write the assumptions down so the review attacks the assumption, not the conclusion, and you can rerun in seconds when an input changes.

## Further reading

- [Turning vague asks into requirements and SLOs](/blog/software-development/system-design/turning-vague-asks-into-requirements-and-slos) — where the inputs to your estimate come from.
- [Articulating trade-offs: CAP, PACELC, and beyond](/blog/software-development/system-design/articulating-tradeoffs-cap-pacelc-and-beyond) — naming the trade-offs your numbers surface.
- [Caching strategies and the pitfalls that bite](/blog/software-development/system-design/caching-strategies-and-the-pitfalls-that-bite) — the main lever for read-heavy systems, done right.
- [Partitioning and sharding without downtime](/blog/software-development/system-design/partitioning-and-sharding-without-downtime) — what to do when the estimate says "doesn't fit on one box."
- [Database partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding) and [consistent hashing and data partitioning](/blog/software-development/database/consistent-hashing-and-data-partitioning) — the mechanisms behind spreading writes.
- [LSM-trees as write-optimized storage engines](/blog/software-development/database/lsm-trees-write-optimized-storage-engines) and [B-trees and how database indexes work](/blog/software-development/database/b-trees-how-database-indexes-work) — why sequential I/O and storage-engine choice follow from your read/write ratio.
- [Kafka as a distributed log](/blog/software-development/database/kafka-as-a-distributed-log) — the throughput constant behind log-based ingest.
- Jeff Dean, "Numbers Everyone Should Know" (and the widely-circulated "Latency Numbers Every Programmer Should Know" interactive charts) — the canonical source for the latency ladder.
- *Designing Data-Intensive Applications* by Martin Kleppmann — the standard reference for the mechanisms these estimates size.
