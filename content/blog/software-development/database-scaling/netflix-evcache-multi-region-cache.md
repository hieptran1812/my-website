---
title: "Netflix EVCache: A Cache That Survives an AZ Going Dark"
date: "2026-06-30"
publishDate: "2026-06-30"
description: "A deep dive into Netflix's EVCache — an AZ-aware, multi-replica memcached layer engineered for warm failover across instance, zone, and region loss, and how its redundancy-first design sits on the opposite end of the axis from Facebook's memory-efficient memcache."
tags: ["caching", "memcached", "distributed-systems", "high-availability", "netflix", "aws", "multi-region", "evcache", "system-design", "database-scaling"]
category: "software-development"
subcategory: "Database Scaling"
author: "Hiep Tran"
featured: true
readTime: 34
---

Most caches are built to make a fast database faster. EVCache is built to make sure a server going up in smoke does not turn into a 2 a.m. incident. That difference in goal — latency optimization versus availability engineering — shows up in every architectural decision Netflix made, and it is the reason their cache looks nothing like the textbook "memcached in front of MySQL" you have probably built.

Here is the operating reality EVCache was designed for. Netflix runs entirely on AWS, spread across multiple availability zones and regions. The home screen, the personalized rows, the "because you watched" recommendations — almost all of it is precomputed offline and served from cache on the hot read path. When a member opens the app, the service does not recompute their recommendations; it reads precomputed data out of a cache and renders. If that cache is cold, the fallback is not "slightly slower" — it is an origin service that was never sized to absorb the full read load, because the entire point of precomputing was to keep it out of the request path. And AWS availability zones do disappear. Sometimes a single instance dies. Sometimes a whole zone degrades. Netflix runs Chaos Monkey and Chaos Kong specifically to make this routine, not exceptional. A cache that loses a meaningful fraction of its hot data when one zone hiccups is, for Netflix, a cache that has failed at its only job.

The diagram below is the mental model for the whole article: the EVCache client writes to a *full copy of the data in every availability zone*, but reads only from its own zone. Hold that single asymmetry in your head — write-everywhere, read-local — and most of EVCache follows from it.

![EVCache write fan-out with local-zone reads across three availability zone server groups](/imgs/blogs/netflix-evcache-multi-region-cache-1.webp)

The client (a library inside the application, not a separate proxy tier) owns the topology. On a write — `set` or `delete` — it fans the mutation out to one server group per availability zone, each of which holds a complete replica of the keyspace. On a read — `get` — it talks only to the server group in its own zone. That keeps reads fast (no cross-zone network hop on the happy path) and keeps every zone independently survivable (each zone has all the data, so losing peers does not create holes). The cost is obvious and deliberate: you are storing the full dataset N times, once per zone. Netflix decided that redundancy was worth the memory. This article is about why, how, and when that is the right trade — and when it is not.

## Why EVCache is different from the cache you have built

Start by being honest about the cache most teams actually run, then line it up against what Netflix needed.

| Assumption | The cache most teams build | The reality EVCache is built for |
| --- | --- | --- |
| Cache misses are cheap | A miss just hits the database, which is fine | A cold cache hits a service sized only for cache-miss trickle, not full load — a self-inflicted outage |
| One copy of cached data is enough | Hash keys across a pool; lose a node, lose its keys | Losing a zone's worth of keys means a thundering herd onto origin; unacceptable on the home-screen path |
| The cache lives in one place | One region, one cluster | Multiple AWS regions; a region can be evacuated and traffic shifted in minutes |
| Consistency is the cache's problem | Wire up invalidation carefully and pray | It is a cache: eventual consistency is acceptable, and TTLs bound staleness |
| Memory is the constraint to optimize | Pack as much into as little RAM as possible | Availability is the constraint; memory cost is a knob you are willing to turn up |

The crucial line is the second one. For a lot of systems, a cache miss is a non-event — the database absorbs it. For Netflix's read path, the database (really, a set of precompute-fed online services) is sized on the assumption that the cache is doing its job. A mass eviction — a whole zone's cache going cold at once — is the failure mode that keeps cache engineers up at night, because it converts a localized hardware blip into a correlated load spike on a tier that cannot take it. EVCache's entire design is a refusal to ever let that happen.

> A cache that is optional is a performance feature. A cache that is load-bearing is an availability system. EVCache is the second kind, and it is engineered accordingly.

If you have read [Scaling memcache at Facebook](/blog/software-development/database-scaling/scaling-memcache-at-facebook), you have seen the same memcached primitive aimed at a *different* constraint. Facebook's memcache fights to be memory-efficient: a regional pool holds roughly one copy of the data, and elaborate machinery — leases, gutter pools, careful invalidation — keeps that one copy correct and stampede-free. EVCache spends memory to buy redundancy instead. Two world-class engineering orgs, the same building block, opposite ends of one axis. We will make that axis explicit later; keep it in the back of your mind as we go.

## 1. The topology: server groups, one full replica per zone

**Senior rule of thumb: in EVCache, an availability zone is a unit of replication, not a unit of sharding.**

This is the single most important thing to internalize, because it inverts the instinct most engineers have. The usual move is "shard the data across all my nodes, everywhere." EVCache shards *within* a zone and *replicates across* zones.

Concretely, an EVCache cluster in a region is made of **server groups**, one per availability zone. Each server group is a set of memcached instances that together hold a complete copy of the keyspace, sharded across those instances using **Ketama consistent hashing**. If a region has three AZs, the cluster has three server groups, and the same key `user:1234:home` lives in all three — once in zone a's server group, once in zone b's, once in zone c's, each on whichever instance Ketama maps it to within that group.

The client maintains a connection to each server group and knows which AZ it itself is running in (instance metadata makes this trivial on EC2; service discovery via Eureka tracks the live instances per group). From there the read/write asymmetry is just policy:

```python
# Illustrative EVCache-style client. The real client is Java (spymemcached
# internally); this captures the write-everywhere / read-local policy.

class EVCacheClient:
    def __init__(self, server_groups, local_zone):
        # server_groups: { "us-east-1a": ServerGroup, "us-east-1b": ..., ... }
        # Each ServerGroup is a Ketama ring over that zone's memcached instances.
        self.server_groups = server_groups
        self.local_zone = local_zone

    def set(self, key, value, ttl_seconds):
        # Fan the write out to EVERY zone's replica. All copies, every time.
        results = []
        for zone, group in self.server_groups.items():
            node = group.ketama_node_for(key)   # consistent-hash within the zone
            results.append(node.set(key, value, ttl_seconds))
        # A write is "good enough" if the local zone took it; peers are async-ish.
        return self._quorum(results, require_local=True)

    def delete(self, key):
        for zone, group in self.server_groups.items():
            group.ketama_node_for(key).delete(key)

    def get(self, key):
        # Read from the LOCAL zone first — no cross-zone hop on the happy path.
        local = self.server_groups[self.local_zone]
        value = local.ketama_node_for(key).get(key)
        if value is not None:
            return value
        # Local miss (eviction, cold instance, or a node down): fall back to a
        # peer zone's full replica. Still a cache hit — we never touch origin.
        return self._zone_fallback(key, skip=self.local_zone)

    def _zone_fallback(self, key, skip):
        for zone, group in self.server_groups.items():
            if zone == skip:
                continue
            value = group.ketama_node_for(key).get(key)
            if value is not None:
                return value          # warm peer replica saved the read
        return None                   # genuine miss → caller hits origin

    def _quorum(self, results, require_local):
        # EVCache treats the local write as the one that must succeed; peer
        # writes failing is logged and retried, not surfaced as an error.
        return results[0] if require_local else all(results)
```

Three properties fall out of this code, and each is load-bearing:

1. **Reads are local and therefore fast.** No cross-AZ round trip on the common path. Cross-zone traffic inside an AWS region is cheap and low-latency, but "low" is still tens to hundreds of microseconds more than a same-zone hop, and at Netflix's read volume that adds up. Keeping the happy path local is a latency decision.
2. **Reads have a warm fallback that is never the database.** A local miss does not go to origin — it goes to a *peer zone's full replica*, which almost certainly has the value, because every write went there too. This is the property that makes a zone failure a non-event for read availability.
3. **Writes are more expensive than reads, on purpose.** Every `set` does N network operations instead of one. EVCache accepts a heavier write path to make the read path bulletproof. For a workload that is overwhelmingly reads of precomputed data, that is exactly the right place to spend.

### Second-order optimization: the local write must succeed, peers can lag

A naive "write to all N, fail if any fails" policy would make your write availability *worse* than a single cache — you would be multiplying failure probability. EVCache does not do that. The local-zone write is the one that gates success; peer-zone writes are best-effort and retried out of band. This is the eventual-consistency posture showing up at the smallest scale: it is a cache, so a peer zone being a few milliseconds behind on one key is fine, and is strictly better than refusing the write.

## 2. The read path: local hit versus cross-zone fallback

**Senior rule of thumb: design the fallback so the worst case is "one extra network hop," never "a database query."**

Let us trace the two read paths side by side, because the difference between them is the whole availability story.

![Read path comparison: a local-zone hit returns in sub-millisecond time while a local miss falls back to a peer zone replica, never to the database](/imgs/blogs/netflix-evcache-multi-region-cache-6.webp)

On the left, the common case: `get(key)` in zone a hashes (via Ketama) to a local memcached instance, finds the value in RAM, and returns in well under a millisecond with no extra hops. This is what happens the overwhelming majority of the time. On the right, the interesting case: the value is *not* in the local zone — maybe that instance evicted it under memory pressure, maybe the instance is down, maybe it briefly fell out of the ring during a deploy. Instead of declaring a miss and punting to origin, the client retries against a peer zone's replica. Because every write fanned out to that zone too, the value is there. The read costs one extra cross-AZ hop and still never touches the database.

That bears repeating because it is the crux: **a local miss in EVCache is usually still a cache hit.** The redundancy converts what would be a database query in a single-copy cache into a slightly-slower cache read. Multiply that across millions of reads per second during a partial zone degradation and you can see why Netflix pays the memory tax. The alternative — a stampede of cache misses onto a precompute-fed service — is the exact correlated-failure scenario that takes down the home screen.

This is also where EVCache's philosophy diverges sharply from a single-pool cache and connects to the broader pattern in [the caching hierarchy at scale](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale): EVCache deliberately keeps the cache miss *inside the cache tier* by adding a horizontal fallback layer (peer zones) before the vertical fallback (origin) is ever considered.

### Second-order optimization: fallback is opt-in per call, because it costs latency

Zone fallback is not free — it adds a hop and consumes a connection slot on a remote group. EVCache makes it configurable. For a request where a cold miss is genuinely cheap (the value is trivially recomputable, or the caller has its own fallback), you may *disable* zone fallback and just take the miss locally. For the home-screen path, where a miss is expensive, you enable it. The design lets the caller decide where on the latency-versus-availability curve each call site should sit, rather than baking one answer in. That is the kind of knob that distinguishes a cache built by people who have operated it from one built to a spec.

## 3. When an availability zone goes dark

**Senior rule of thumb: the test of an availability design is not whether it survives a node failure — it is whether it survives a *correlated* failure of everything in one fault domain.**

A single dead instance is easy; consistent hashing already handles it (we will get to that). The hard case is an entire zone degrading — the canonical AWS gray failure where one AZ's network or control plane goes sideways and a third of your fleet becomes unreachable at once. This is exactly what Chaos Kong simulates, and it is the scenario EVCache is built to shrug off.

The animation below shows what happens to reads when zone a goes dark. In the healthy state, the client in zone a reads from zone a's local replica. When zone a goes dark, those local reads start failing — and the client fails them over to zone b's full replica. No cold-cache stampede, no origin overload: the reads simply re-home to a warm copy in another fault domain.

<figure class="blog-anim">
<svg viewBox="0 0 720 320" role="img" aria-label="An availability zone goes dark and reads fail over to a peer zone's warm replica" style="width:100%;height:auto;max-width:820px">
<style>
.ev-box{fill:var(--surface,#f3f4f6);stroke:var(--border,#d1d5db);stroke-width:2}
.ev-client{fill:var(--accent,#6366f1);opacity:.16;stroke:var(--accent,#6366f1);stroke-width:2}
.ev-lbl{font:600 15px ui-sans-serif,system-ui;fill:var(--text-primary,#1f2937);text-anchor:middle}
.ev-sub{font:500 12px ui-sans-serif,system-ui;fill:var(--text-secondary,#6b7280);text-anchor:middle}
.ev-path{stroke:var(--accent,#6366f1);stroke-width:4;fill:none}
.ev-dead{fill:#ef4444;opacity:0}
.ev-deadlbl{font:700 13px ui-sans-serif,system-ui;fill:#ef4444;text-anchor:middle;opacity:0}
@keyframes ev-zoneA{0%,40%{opacity:1}55%,95%{opacity:.12}100%{opacity:1}}
@keyframes ev-killmark{0%,40%{opacity:0}55%,95%{opacity:1}100%{opacity:0}}
@keyframes ev-readA{0%,40%{opacity:1}55%,100%{opacity:0}}
@keyframes ev-readB{0%,45%{opacity:0}60%,95%{opacity:1}100%{opacity:0}}
.ev-aliveA{animation:ev-zoneA 9s ease-in-out infinite}
.ev-mark{animation:ev-killmark 9s ease-in-out infinite}
.ev-toA{animation:ev-readA 9s ease-in-out infinite}
.ev-toB{animation:ev-readB 9s ease-in-out infinite}
@media (prefers-reduced-motion:reduce){.ev-aliveA{animation:none;opacity:.12}.ev-mark{animation:none;opacity:1}.ev-toA{animation:none;opacity:0}.ev-toB{animation:none;opacity:1}}
</style>
<rect class="ev-client" x="40" y="120" width="150" height="80" rx="10"/>
<text class="ev-lbl" x="115" y="155">client</text>
<text class="ev-sub" x="115" y="176">in AZ-a</text>
<rect class="ev-box ev-aliveA" x="430" y="30" width="240" height="80" rx="10"/>
<text class="ev-lbl ev-aliveA" x="550" y="62">server group AZ-a</text>
<text class="ev-sub ev-aliveA" x="550" y="84">local replica</text>
<rect class="ev-dead" x="430" y="30" width="240" height="80" rx="10"/>
<text class="ev-deadlbl ev-mark" x="550" y="78">AZ-a DARK</text>
<rect class="ev-box" x="430" y="210" width="240" height="80" rx="10"/>
<text class="ev-lbl" x="550" y="242">server group AZ-b</text>
<text class="ev-sub" x="550" y="264">warm replica</text>
<path class="ev-path ev-toA" d="M190 150 L430 90"/>
<path class="ev-path ev-toB" d="M190 170 L430 250"/>
<text class="ev-sub ev-toA" x="300" y="105">read local</text>
<text class="ev-sub ev-toB" x="300" y="232">fail over, no cold miss</text>
</svg>
<figcaption>When AZ-a goes dark, the client retries AZ-b's full replica; reads stay warm instead of stampeding the database.</figcaption>
</figure>

Compare this to a single-pool cache. There, a zone's worth of cache nodes vanishing means a zone's worth of keys vanish; every read for those keys becomes a miss, and every miss becomes an origin query, all at once. That is the thundering herd — the failure mode dissected in [cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd). EVCache does not have a herd to thunder, because the data was never gone; it was sitting in the other zones the whole time.

The recovery story matters too. When zone a comes back, its server group starts cold — fresh instances with empty caches. In a single-copy design, that cold tier would either serve misses (bad) or have to be warmed from the database (load spike, slow). EVCache instead lets the recovering zone refill organically: writes have been flowing to it all along once instances rejoin (the client fans every write to all live groups), and reads that land there and miss fall back to a warm peer and can be re-populated. Netflix has also built dedicated cache-warming tooling for the harder version of this problem — bulk-moving cache state between instances using EBS volumes rather than re-reading from origin — so that bringing a large cluster back does not depend on the origin tier at all.

### Second-order optimization: replica count is a dial, not a constant

Three zones is the common shape, but "one replica per AZ" generalizes. The number of copies you keep is a cost-versus-availability dial: more replicas mean more memory and more write amplification but stronger survivability and lower fallback latency (more nearby warm copies). For data where a cold start is catastrophic, you keep a copy everywhere. For data that is cheap to recompute, you might keep fewer. EVCache lets different caches make different choices, which is the whole point of treating redundancy as a knob.

## 4. Ketama hashing: why losing one instance is not a crisis

**Senior rule of thumb: never use `hash(key) % N` for a distributed cache — the day N changes, you lose almost everything.**

Inside a single server group, keys are distributed across instances. The naive way to do this is modulo hashing: pick the instance with `hash(key) % N`. It works perfectly until the instant `N` changes — one instance dies, or you scale up — at which point nearly every key maps to a *different* instance, and your cache effectively flushes itself. For a load-bearing cache, a self-inflicted full flush is indistinguishable from an outage.

![Modulo hashing reshuffles nearly every key when the node count changes, while Ketama consistent hashing moves only the dead node's share of keys](/imgs/blogs/netflix-evcache-multi-region-cache-7.webp)

Consistent hashing — Ketama is a specific, widely-used implementation — fixes this. Keys and instances are both hashed onto a ring; a key belongs to the next instance clockwise from its position. When an instance dies, only the keys in *its* arc of the ring re-home, to the next instance clockwise. Everything else stays put. Instead of re-mapping ~100% of keys on a node change, you re-map roughly `1/N` of them. With many virtual nodes per physical instance (which Ketama uses to smooth the distribution), the reshuffled fraction is small and evenly spread, so no single surviving instance gets slammed.

```python
import bisect, hashlib

class KetamaRing:
    def __init__(self, instances, vnodes=160):
        # vnodes: virtual nodes per physical instance, to smooth the ring.
        self.ring = {}          # hash position -> instance
        self.sorted_keys = []
        for inst in instances:
            for v in range(vnodes):
                h = self._hash(f"{inst}#{v}")
                self.ring[h] = inst
        self.sorted_keys = sorted(self.ring)

    def _hash(self, s):
        return int(hashlib.md5(s.encode()).hexdigest(), 16)

    def node_for(self, key):
        h = self._hash(key)
        # First ring position clockwise from the key's hash.
        i = bisect.bisect(self.sorted_keys, h) % len(self.sorted_keys)
        return self.ring[self.sorted_keys[i]]

# Lose one of five instances: only that instance's arc re-homes.
ring = KetamaRing(["i-a", "i-b", "i-c", "i-d", "i-e"])
hot_key = "user:1234:home"
print(ring.node_for(hot_key))   # e.g. "i-c"
# Rebuild without i-c: keys that landed on i-c now go to the next node
# clockwise; the other ~4/5 of keys are unaffected.
```

The reason this matters for EVCache specifically is that it composes with the cross-zone redundancy. When an instance dies inside zone a's server group, Ketama re-homes its arc within zone a — but during the brief window before the ring settles and the new owner is populated, the keys on that arc would miss locally. That local miss falls back to a peer zone, which still has the data. So Ketama handles *intra-zone* node loss gracefully, and zone fallback handles the residual misses. The two mechanisms stack: consistent hashing minimizes the blast radius of a single instance, and replication catches whatever leaks through.

### Second-order optimization: virtual nodes and the hot-arc problem

A subtle failure mode of consistent hashing is uneven arcs: with few points on the ring, one instance can own a disproportionately large arc and become a hotspot. Ketama's use of many virtual nodes per instance (commonly 100–200) keeps arc sizes roughly uniform, so load and the re-homing blast radius are both evenly spread. If you ever roll your own consistent hashing and skip virtual nodes, you will discover this the hard way when one instance runs hot for no apparent reason.

## 5. The global cache: cross-region replication

**Senior rule of thumb: if you can fail traffic over to another region in minutes, that region's cache had better already be warm — warming it during the failover is too late.**

Netflix does not just run in multiple availability zones; it runs in multiple AWS regions and can evacuate an entire region, shifting that region's traffic to others. This is the regional-failover capability that makes Netflix resilient to a whole-region event. But it creates a caching problem: if you move millions of members' traffic to a region whose EVCache is cold, you have just recreated the thundering-herd problem at continental scale. The receiving region's precompute-fed services would be flattened by the miss storm.

The fix is to keep every region's cache warm *before* it is ever needed, by replicating mutations across regions. Netflix calls the result the **global cache**, and the project that built it is named **Moneta** (after the Roman goddess associated with memory). The replication is integrated into the EVCache client and a supporting pipeline rather than bolted on as an afterthought.

![Cross-region replication pipeline: a mutation in one region flows through a replication relay and Kafka to a remote consumer that re-fetches the value and applies it, keeping the peer region warm](/imgs/blogs/netflix-evcache-multi-region-cache-4.webp)

The flow, left to right and top to bottom in the figure: a `set` or `delete` in the source region (say `us-east-1`) first writes to all of that region's AZ server groups, as always. The mutation is also captured and handed to a **replication relay**, which reads the stream of mutations and publishes them to a **Kafka** topic — Netflix's replication pipeline has historically been Kafka-based. A **consumer in the remote region** (say `eu-west-1`) reads that stream. Crucially, what travels across the region boundary is the *mutation event* — the key and metadata about what changed — and the remote side then **re-fetches the actual value from the source region** before applying it to its own server groups. (Shipping the key and re-reading the value, rather than streaming full payloads across regions, keeps the cross-region pipe lean and sidesteps a class of ordering and size problems with large values.) The net effect: the remote region's cache stays populated with current data, so when a regional failover lands traffic on it, it is **warm**.

Here is a sketch of the producer and consumer halves, emphasizing TTL handling and the re-fetch:

```python
# Producer side, in the source region: every mutation is mirrored to Kafka.

def evcache_set_replicated(client, key, value, ttl_seconds):
    client.set(key, value, ttl_seconds)          # local: fan out to all AZs
    replication_relay.publish({                   # async: mirror to peer regions
        "op": "set",
        "key": key,
        "ttl": ttl_seconds,
        "source_region": REGION,
        "ts": now_ms(),
    })                                            # NOTE: key + metadata, not the value

# Consumer side, in a remote region: re-fetch the value, then apply locally.

def apply_remote_mutation(local_client, event):
    if event["op"] == "delete":
        local_client.delete(event["key"])
        return
    # Re-read the current value from the source region's cache. Re-fetching
    # (rather than trusting a streamed payload) lets last-write-win on a
    # late/duplicate event and keeps large values off the cross-region pipe.
    value = source_region_client(event["source_region"]).get(event["key"])
    if value is None:
        return                                    # already evicted/expired; skip
    # Carry the remaining TTL so the remote copy expires roughly in step.
    remaining = max(1, event["ttl"] - (now_ms() - event["ts"]) // 1000)
    local_client.set(event["key"], value, remaining)
```

### Eventual consistency, and why it is fine

Cross-region replication is asynchronous, so there is **replication lag** — typically small (the pipeline is built for low latency at very high event volume; Netflix has reported tens of millions of replication events flowing globally), but non-zero. For a window after a write in `us-east-1`, a reader in `eu-west-1` may see a stale value or no value. EVCache accepts this. It is a cache, the source of truth lives elsewhere, and TTLs bound how long any staleness can persist. The system is explicitly *eventually consistent*, and that relaxation is exactly what buys the availability and the cheap, warm cross-region state. If you needed strong cross-region consistency from your cache, EVCache would be the wrong tool — but you almost never do, and pretending you do is how people accidentally build a slow, fragile, globally-synchronous database and call it a cache.

This is the same lesson that shows up whenever you put replicas in the read path; see [read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas) for the general version of "asynchronous replicas trade a bounded staleness window for read availability and scale."

### Second-order optimization: replicate invalidations promptly, tolerate stale data lazily

A subtle but important asymmetry: a stale *value* in a remote region is usually harmless (the next write or the TTL fixes it), but a *missed delete* can be actively wrong — serving content that should have been pulled. So the replication pipeline treats deletes and invalidations as first-class, ordered events, while being relaxed about the freshness of values. Getting this asymmetry right is the difference between "occasionally a few seconds stale" (fine) and "occasionally serving data we promised to remove" (not fine).

## 6. Moneta and RocksDB: holding more than RAM can afford

**Senior rule of thumb: when your cache is mostly cold long-tail data, paying for RAM to hold all of it is a waste — push the tail onto SSD.**

Full per-AZ replication is wonderful for availability and brutal on the RAM bill. If you keep N copies of a large keyspace, you are buying N times the memory, and a lot of that memory is holding cold, rarely-read items that happen to need to exist for the rare request. Netflix's response — part of the same Moneta project — was to stop treating each EVCache instance as a single in-memory store and turn it into a **two-level cache**: a small hot set in RAM, a large long tail on local SSD.

![Moneta two-level cache: the Rend Go proxy fronts an L1 memcached RAM cache and an L2 Mnemonic/RocksDB SSD cache inside a single EC2 instance](/imgs/blogs/netflix-evcache-multi-region-cache-3.webp)

Reading the stack top to bottom: the **EVCache client** speaks the ordinary memcached protocol, unaware that anything has changed. On the instance, a Go proxy called **Rend** terminates that protocol and owns the policy for two storage tiers beneath it. **L1** is a memcached process holding the hot subset in RAM — fast, small. **L2** is **Mnemonic**, a component backed by **RocksDB** on the instance's local SSD, holding the larger-than-RAM long tail at a far lower cost per gigabyte. Rend decides what lives where, serves reads from L1 when it can and L2 when it must, and manages the data layout across the two tiers. The whole thing runs on an EC2 instance type deliberately chosen with *less RAM and much more SSD* than the all-in-memory configuration would need.

The economics are the entire point. RocksDB is an LSM-tree key-value store designed for exactly this: high write throughput and acceptable read latency on SSD. By moving the cold tail from expensive RAM to cheap SSD, Netflix keeps the same item count and the same per-AZ redundancy at a materially lower hardware cost. The hot items the request path actually touches still come from RAM, so the typical read stays fast; only the rare long-tail read pays the SSD latency. It is the classic memory hierarchy — fast-and-small over slow-and-large — applied to a distributed cache instead of a CPU.

```
get(key) on a Moneta instance:
  Rend receives the memcached get
  -> check L1 (memcached, RAM)
        hit:  return immediately  (the common case for hot keys)
        miss: check L2 (Mnemonic -> RocksDB, SSD)
                 hit:  return (and optionally promote into L1)
                 miss: report miss to the client
                       (client then tries zone fallback, as in section 2)
```

Note how cleanly this composes with everything above: Moneta is purely an *intra-instance* optimization. The client still fans writes to every zone, still reads local-first with zone fallback, still replicates cross-region. Rend's L1/L2 split is invisible above the memcached protocol. Layering a storage optimization underneath an unchanged protocol boundary is exactly why this could be rolled out without rewriting the clients — a textbook example of the value of a stable interface.

### Second-order optimization: not every cache wants two levels

Moneta is a cost play, and it pays off when the keyspace is large and the access pattern is skewed (a hot head, a long cold tail) — which describes a lot of Netflix's precomputed data. For a small, uniformly-hot cache, adding an SSD tier just adds latency and complexity for no memory savings, because everything is hot anyway. EVCache supports both shapes; the two-level instance is an option you reach for when the RAM bill justifies it, not a default.

## 7. EVCache as a primary store for ephemeral data

**Senior rule of thumb: a cache reliable enough to be the only copy of *short-lived* data is a different, more demanding thing than a cache that just accelerates a database.**

Most caches are strictly an accelerator: lose the cache and the truth still lives in the database. EVCache's redundancy makes it reliable enough that Netflix sometimes uses it as the *primary* store for **ephemeral data** — data that is inherently short-lived and does not need to outlive its TTL, so there is no separate database of record behind it. Think transient session-like state, in-flight playback bookkeeping, short-horizon counters: data that must be highly available for its lifetime but is meaningless after.

This works precisely because of the full per-AZ replication. If EVCache kept one copy, using it as a primary store would be reckless — an instance loss would be data loss. With a full replica in every zone, a single instance or even a whole zone failing does not lose the data; it is still in the peer zones. The redundancy that was bought for *availability of cached database data* turns out to also buy *durability-for-a-TTL of primary data*. That is a second payoff from the same architectural decision, and it is why "EVCache is just a cache" undersells what the design enables.

The discipline required is to be honest about the word *ephemeral*. EVCache is not a database; it has no on-disk write-ahead log guaranteeing durability across a total loss, no point-in-time recovery, no strong consistency. Using it as a primary store is appropriate only when the data's value is bounded by its TTL and an extraordinarily rare total-cluster loss is an acceptable (and recoverable-by-recompute or simply-tolerable) event. Stretch it past that and you have talked yourself into using a cache as a database, which ends badly.

### Second-order optimization: TTLs are your durability contract

When EVCache is a primary store, the TTL is not just an eviction hint — it is the explicit statement of how long the data is allowed to matter. Setting it correctly is a correctness decision, not a tuning one. Too short and you lose data that callers still need; too long and you accumulate state that should have evaporated. The TTL is the contract, so write it down and mean it.

## 8. The redundancy-versus-memory axis: EVCache versus Facebook memcache

**Senior rule of thumb: there is no universally correct cache design — there is a position on the redundancy-versus-memory axis, and you pick yours from your failure economics.**

Now we can make the comparison explicit, because it is the most instructive thing in this whole story. Facebook's memcache (covered in [scaling memcache at Facebook](/blog/software-development/database-scaling/scaling-memcache-at-facebook)) and Netflix's EVCache are built on the *same* primitive — memcached — and arrive at nearly opposite architectures because they optimized for opposite constraints.

![A comparison matrix of Facebook memcache versus Netflix EVCache across copies of data, read path, behavior on AZ loss, consistency tooling, and what each optimizes for](/imgs/blogs/netflix-evcache-multi-region-cache-5.webp)

| Dimension | Facebook memcache | Netflix EVCache |
| --- | --- | --- |
| Copies of the data | Roughly one regional pool (plus replica pools where read load demands) | One full replica per availability zone |
| Read path | Hash to the pool; read from whichever host owns the key | Read the local-zone replica first; fall back cross-zone on a miss |
| Behavior on losing a fault domain | Lost keys become misses → load hits the database | Reads fail over to a warm peer replica; database is untouched |
| Primary consistency tool | Leases and careful invalidation to prevent stampedes on the single copy | Eventual consistency, TTLs, and cross-region replication |
| Write cost | One write to the pool | N writes, one per zone (write amplification) |
| Memory cost | Low — close to one copy of the data | High — N copies of the data |
| What it optimizes for | **Memory efficiency** | **Availability and warm failover** |

Read the bottom row as the thesis. Facebook spends engineering effort to be *memory-efficient*: keep about one copy, and build sophisticated mechanisms (leases to coordinate concurrent fills, gutter pools to absorb dead servers, tight invalidation) to make that one copy correct and stampede-resistant. EVCache spends *memory* to be *redundant*: keep a full copy per zone, and let the redundancy itself provide most of the availability, so the consistency machinery can be simpler (TTLs and eventual consistency) because no single copy is load-bearing.

Neither is wrong. They are answers to different questions:

- Facebook's question was roughly: *we have an enormous working set and want to serve it from the least RAM possible while staying correct under brutal concurrency.* The scarce resource is memory, so the design hoards it.
- Netflix's question was roughly: *our read path is load-bearing and an AWS fault domain can vanish without warning; we cannot let a zone loss become an origin stampede.* The scarce resource is availability, and memory is something they are willing to spend to get it.

If you take one transferable idea from this article, it is this axis. When you design a cache, do not ask "what is the best cache architecture?" Ask "how expensive is a cache miss for me, and how correlated are my failures?" If misses are cheap and failures uncorrelated, lean toward Facebook's memory efficiency. If misses are catastrophic and failures come in fault-domain-sized chunks, lean toward EVCache's redundancy. Most real systems sit somewhere on the line, and knowing the line exists is most of the battle.

> Facebook and Netflix did not disagree about how memcached works. They disagreed about what was scarce. Architecture is just the shadow that scarcity casts.

## Case studies from production

The following are illustrative incidents — composite scenarios drawn from the public behavior and design of EVCache and the failure modes the architecture is built to handle. They are written the way these things actually unfold on call.

### 1. The zone that vanished at dinner time

The symptom: a single AWS availability zone in the primary region started returning elevated connection errors during peak evening streaming hours. The first hypothesis was a bad deploy in the affected zone's services. The actual root cause was an AZ-level networking degradation — a textbook gray failure, nothing the team had shipped. What was striking was the *non-event* in the cache tier: home-screen reads in the degraded zone failed locally and immediately fell over to peer-zone replicas. Origin services saw no read spike, because the misses never reached them. The fix was, mostly, to wait for AWS to recover the zone while traffic re-homed. The lesson: the architecture had already paid for this. The redundancy that looked like a wasteful 3x memory bill on the spreadsheet was, that evening, the only reason a routine cloud-provider blip did not become a streaming outage.

### 2. The modulo-hashing regression

The symptom: after a library change in a non-EVCache internal cache, a small service saw its cache hit rate collapse to near zero every time it autoscaled. The wrong first hypothesis was a TTL misconfiguration. The actual root cause: the service had quietly switched its client-side sharding to `hash(key) % N`, so every scale-up event changed `N` and re-mapped essentially the entire keyspace, flushing the cache on every deploy and scaling action. The fix was to move to consistent hashing (Ketama, matching EVCache's own approach), after which a node change re-homed only `~1/N` of keys. The lesson: this is the exact trap EVCache avoids by construction inside each server group, and it is so easy to fall into that even teams who *know* better re-introduce it whenever someone hand-rolls sharding.

### 3. The cross-region failover that landed cold

The symptom (in a counterfactual where the global cache had not yet existed): an entire region was evacuated, traffic shifted to a peer region, and that region's precompute-fed services immediately saturated. The root cause was a cold cache in the receiving region — every read was a miss, every miss hit origin, and origin was sized for steady-state, not for absorbing a continent's worth of cold reads at once. This is precisely the scenario that motivated the global cache. Once cross-region replication kept the peer region warm, the same failover became survivable: reads in the receiving region were hits because mutations had been flowing in continuously. The lesson: regional failover is only as good as the warmth of the region you fail *to*. Warming on demand is too late; you warm continuously, in advance, or you do not really have a failover.

### 4. The RAM bill that did not add up

The symptom: a large EVCache cluster's cost was dominated by memory, yet instrumentation showed the vast majority of items were read rarely or never within their TTL. The wrong first instinct was to shrink the keyspace or shorten TTLs, both of which would have hurt hit rates. The actual insight: the access pattern was a small hot head over a very long cold tail, and the cold tail did not need to be in RAM at all. The fix was Moneta — move to two-level instances with a memcached L1 in RAM and a RocksDB-backed L2 on SSD, fronted by Rend, on instance types with less RAM and more SSD. Item count and per-AZ redundancy were preserved; cost dropped substantially because the cold tail moved to cheap storage. The lesson: full replication and a large RAM footprint are separable problems. You can keep the redundancy and still get the cold data off RAM.

### 5. The over-eager write fan-out

The symptom: a write-heavy use case (mistakenly) put on EVCache showed surprisingly high write latency and load. The wrong hypothesis was network congestion. The actual root cause was inherent: every `set` fanned out to N zones, so a write-heavy workload was doing N times the write work, and the local-write-must-succeed semantics meant latency was bounded by the local write while peer writes consumed connections and bandwidth. The fix was recognizing the workload was a poor fit — EVCache is tuned for read-heavy precomputed data, and a write-dominated workload pays the replication tax on every operation with little benefit. The lesson: the write amplification that is a non-issue for a read-heavy cache becomes the dominant cost for a write-heavy one. Match the tool to the read/write ratio.

### 6. The missed delete

The symptom: in a cross-region setup, a piece of content that had been removed in the source region was briefly still served in a remote region. The wrong hypothesis was a stale-value bug in value replication. The actual root cause was an ordering/lag issue specifically affecting a delete event in the replication pipeline — the value had been correctly replicated earlier, but the subsequent delete was delayed, so the remote region kept serving the now-invalid value until its TTL expired. The fix was to treat invalidations and deletes as first-class, promptly-ordered events in the pipeline, distinct from the more relaxed handling of value freshness. The lesson: in an eventually-consistent cache, a stale value is usually benign but a missed delete can be a correctness or compliance problem. Replicate the removals faster than you replicate the additions.

### 7. The instance that fell out of the ring during a deploy

The symptom: during a rolling deploy of an EVCache server group, hit rate dipped briefly for a subset of keys. The wrong hypothesis was that the deploy was evicting data. The actual root cause was benign and expected: as instances cycled, their arcs of the Ketama ring momentarily had no owner or a freshly-empty owner, so keys on those arcs missed locally for a few seconds. The saving grace was zone fallback — those local misses were served from peer zones, so end users saw no errors, only a transient internal hit-rate wobble. The fix was nothing; the system behaved as designed. The lesson: consistent hashing plus zone fallback turns the routine churn of deploys into invisible internal noise rather than user-visible misses. The two mechanisms exist precisely so that ordinary operations are non-events.

### 8. The "let's add strong consistency" proposal

The symptom: a team wanted read-your-writes consistency across regions from EVCache for a feature where users edited a setting and expected to immediately see it everywhere. The wrong instinct was to make EVCache synchronously replicate across regions to satisfy this. The actual right move was to recognize that imposing synchronous cross-region writes would destroy the very properties — availability, low write latency, partition tolerance — that made EVCache valuable, effectively turning it into a slow, globally-coupled database. The fix was to handle the read-your-writes requirement at the application layer for that specific feature (read from the source region, or pin the user's session) while leaving EVCache eventually consistent for everyone else. The lesson: do not retrofit strong consistency onto a system whose entire value proposition is the relaxation of it. If you need strong cross-region consistency, you need a different tool for that path — not a re-engineered cache.

## When to reach for the EVCache model, and when not to

The EVCache *pattern* — full per-AZ replication, local reads with cross-zone fallback, client-owned topology, eventual consistency, optional cross-region warming and SSD tiering — is not Netflix-specific. It is a reusable design point. The decision tree below sketches when it fits.

![Decision tree for when EVCache's full per-AZ replication model fits a workload versus when a leaner cache is the better choice](/imgs/blogs/netflix-evcache-multi-region-cache-8.webp)

### Reach for the EVCache model when

- **Your read path is load-bearing and a cold cache is catastrophic.** If a mass cache miss would stampede a tier that cannot absorb it, full replication's warm fallback is worth the memory.
- **Your failures come in fault-domain-sized chunks.** Cloud AZs and regions fail as units. If your blast radius is "a whole zone," design for losing a whole zone — which means having the data elsewhere already.
- **You can fail traffic across zones or regions** and need the destination to be warm on arrival. Cross-region replication earns its keep exactly here.
- **The workload is overwhelmingly reads of precomputed or slow-to-derive data.** The write amplification is cheap when writes are rare relative to reads.
- **You want a highly available store for short-lived data with no separate database of record.** Per-AZ replication makes a cache reliable enough to be the primary store for genuinely ephemeral data.
- **The keyspace is large with a skewed access pattern** — then the Moneta two-level (RAM + SSD) instance keeps the redundancy affordable.

### Skip it (or use a leaner cache) when

- **Cache misses are genuinely cheap.** If origin can absorb a miss storm comfortably, you are paying N times the memory to solve a problem you do not have.
- **The workload is write-heavy.** Fan-out write amplification turns from a rounding error into the dominant cost. A single-copy cache or a different store will serve you better.
- **You are in a single zone or single region** with no failover story. Replicating across fault domains you do not have buys nothing.
- **You need strong consistency**, especially across regions. EVCache is deliberately eventually consistent; forcing synchrony onto it discards its reason for existing. Use a system designed for consistency on that path.
- **Your dataset is small and uniformly hot.** The SSD tier adds latency for no savings, and even the full replication may be overkill if a cold start is trivial.

The through-line, one more time: EVCache is what a cache looks like when you treat it as an availability system rather than a performance feature, on infrastructure where fault domains vanish without warning, for a read path you cannot afford to let go cold. Facebook's memcache is what the same primitive looks like when memory is the scarce resource instead. Both are correct. The skill is not memorizing either architecture — it is recognizing which scarcity you are actually facing, and letting that, not habit, choose your position on the redundancy-versus-memory axis.

## Further reading

- Netflix Technology Blog — "Announcing EVCache" and "Ephemeral Volatile Caching in the Cloud" (the original design and the ephemeral-store use case).
- Netflix Technology Blog — "Caching for a Global Netflix" (the global cache, cross-region replication, and the Moneta project with Rend, Mnemonic, and RocksDB).
- Netflix Technology Blog — "Cache Warming: Leveraging EBS for Moving Petabytes of Data" (warming recovering clusters without hitting origin).
- The `Netflix/EVCache` project and wiki (server-group topology, Ketama hashing, zone fallback).
- Sibling posts on this blog: [scaling memcache at Facebook](/blog/software-development/database-scaling/scaling-memcache-at-facebook), [the caching hierarchy at scale](/blog/software-development/database-scaling/the-caching-hierarchy-at-scale), [read scaling with replicas](/blog/software-development/database-scaling/read-scaling-with-replicas), and [cache invalidation and the thundering herd](/blog/software-development/database-scaling/cache-invalidation-and-the-thundering-herd).
