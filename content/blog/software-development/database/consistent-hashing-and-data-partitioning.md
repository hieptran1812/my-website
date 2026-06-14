---
title: "Consistent Hashing and How Distributed Databases Partition Data"
date: "2026-06-14"
publishDate: "2026-06-14"
description: "A deep dive into how distributed databases spread keys across nodes: why naive modulo hashing is catastrophic at scale, how consistent hashing, virtual nodes, and preference lists fix it, and how Dynamo, Cassandra, and Riak put it into production."
tags:
  [
    "consistent-hashing",
    "partitioning",
    "sharding",
    "distributed-systems",
    "virtual-nodes",
    "dynamo",
    "cassandra",
    "hash-ring",
    "rebalancing",
    "databases",
    "rendezvous-hashing",
    "system-design",
  ]
category: "software-development"
subcategory: "Distributed Systems"
author: "Hiep Tran"
featured: true
readTime: 51
image: "/imgs/blogs/consistent-hashing-and-data-partitioning-1.webp"
---

There is a specific failure that almost every team building on a distributed datastore eventually walks into, usually at the worst possible time. The cluster is healthy, traffic is climbing, and someone adds a node to keep up. The dashboards should celebrate: more CPU, more RAM, more disk. Instead the system falls over. Tail latency triples, the cache hit rate craters from 95% to single digits, the origin databases behind the cache get hammered by a stampede of misses, and the on-call engineer is staring at a graph that says throughput went *down* the instant capacity went *up*. The node that was supposed to help is the node that broke everything.

The root cause is almost always the same, and it is almost always invisible until that moment: the system decided which node owns which key using `hash(key) % N`, where `N` is the number of nodes. That single line of code is correct, fast, and a time bomb. Change `N` by one and the arithmetic reshuffles nearly every key to a different node at once. Every cached value is suddenly in the wrong place. Every read misses. The whole cluster spends the next several minutes re-fetching and re-replicating data it already had, while users wait.

The fix the industry converged on is **consistent hashing**, and the data structure at its heart is a ring. The diagram above is the mental model for this entire article: both the servers and the keys are hashed onto one circular keyspace, and a key belongs to the first node you hit walking clockwise from where the key landed. Adding or removing a node moves only the small slice of keys between two neighbors instead of all of them. That one idea — and the engineering scaffolding around it: virtual nodes, preference lists, bounded loads, and a handful of clever alternatives — is what lets Dynamo, Cassandra, Riak, ScyllaDB, and a generation of caches and load balancers grow and shrink without a global reshuffle. This is a tour of how it works, why each piece exists, where it still bites, and how to choose a partitioning scheme for your own system.

## The mental model: one ring, two kinds of dots

![The hash ring: nodes and keys hash onto one circular keyspace, and a key is owned by the first node clockwise from it](/imgs/blogs/consistent-hashing-and-data-partitioning-1.webp)

Read the ring above clockwise. The whole output range of a hash function — say the integers from 0 to 2^160 − 1 for a 160-bit hash like SHA-1 — is bent into a circle so that the largest value wraps around to meet 0. We place two kinds of things on that circle. **Nodes** (the blue boxes) are hashed by their identity — a hostname, an IP, an instance ID — and land at fixed positions. **Keys** (the yellow dots) are hashed the same way and land wherever their hash falls. To find which node owns a key, you start at the key's position and walk clockwise until you hit the first node. That node owns the key. In the figure, key `u:42` lands just before node B, so B owns it; key `u:08` lands just before node E, so E owns it.

That is the entire lookup rule, and almost every desirable property falls out of it. Ownership is a contiguous *arc* of the ring: node B owns everything from the position just after node A up to and including node B's own position. The arcs tile the ring with no gaps and no overlaps, so every possible key has exactly one owner. Crucially, the assignment is *stable under change*. If a node disappears, only its arc needs a new owner, and the natural new owner is the next node clockwise, which simply absorbs the orphaned arc. If a node appears, it carves a new arc out of its clockwise successor and nobody else is affected. Compare that to modulo hashing, where changing the node count rewrites the formula for *every* key. The ring localizes change; modulo globalizes it. That difference is the whole ballgame.

> Consistent hashing is the realization that you should hash the *servers*, not just the keys, and let geometry — not arithmetic — decide ownership. Once both live on the same ring, change becomes local instead of global.

This is, incidentally, exactly the topic of Chapter 6 of Martin Kleppmann's *Designing Data-Intensive Applications*, which treats partitioning (or "sharding," the same idea under a different name) as one of the load-bearing problems of distributed data. Kleppmann goes out of his way to point out that the term "consistent hashing" is *confusingly named*: the word "consistent" here has nothing to do with the consistency of replication or the C in CAP. It refers narrowly to a hash function whose mapping changes minimally when its range changes — Karger's original 1997 sense. Kleppmann notes that in practice many systems that use hash-based partitioning do not use Karger-style consistent hashing at all (Cassandra's documentation, for instance, describes its own scheme in those terms loosely), so the name is best treated as historical jargon rather than a precise description. Keep that caveat in your pocket; we will see why it matters when we get to fixed-partition rebalancing.

## 1. The partitioning problem, stated precisely

**A senior rule of thumb: the partitioner is a function, and you should be able to write its type signature before you write its body.** Partitioning is the act of splitting a dataset too big or too busy for one machine across many machines, each holding a subset called a *partition* or *shard*. Two requirements pull in opposite directions, and every scheme is a compromise between them.

The first requirement is **balance**. If the data and the traffic are not spread evenly, you have not really scaled — you have just moved the bottleneck. A partitioner that puts 90% of the keys on one node has given you a ten-node cluster with the throughput of one node and the operational complexity of ten. The pathological version is a *hot spot*: a single partition that receives a disproportionate share of requests, saturating one node while the rest idle.

The second requirement is **cheap lookup**. Given a key, every client and every node must be able to compute which partition owns it without a network round-trip to a central directory. A coordinator service that you must consult on every request reintroduces the single point of failure and the latency tax you were trying to escape. The best partitioners are pure functions of the key and the cluster membership: `owner = f(key, members)`, computable locally in microseconds.

There are broadly two families of `f`. **Range partitioning** assigns contiguous ranges of the key space to partitions — keys `a` through `m` on node 1, `n` through `z` on node 2. It makes range scans trivial (a scan of `c` through `f` touches one node) but invites hot spots whenever keys arrive in order, because all the new writes pile onto whichever node owns the high end of the range. **Hash partitioning** runs the key through a hash function first and partitions on the hash, which scatters even adjacent keys uniformly and so spreads load beautifully — at the cost of destroying any notion of order, so a range scan must hit every partition. Consistent hashing is a *hash* partitioning scheme; it inherits hashing's even spread and its loss of range scans. We will revisit that tradeoff in depth in section 7, and it is the same tradeoff explored from the schema side in [partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding).

For the rest of this article, keep the type signature in mind. Everything — modulo, the ring, vnodes, rendezvous, jump hash — is a different implementation of `owner = f(key, members)` that tries to keep balance high and the cost of *changing* `members` low.

## 2. The naive answer and why it detonates at scale {#naive-modulo}

**The rule of thumb every distributed-systems engineer learns the hard way: never let your data placement depend on the *count* of nodes.** The most obvious partitioner is modulo. Hash the key to an integer, take it modulo the number of nodes, and that is the partition index.

```python
import hashlib


def hash_key(key: str) -> int:
    """Stable 64-bit integer hash of a key (md5 truncated, deterministic across runs)."""
    digest = hashlib.md5(key.encode()).digest()
    return int.from_bytes(digest[:8], "big")


def modulo_owner(key: str, num_nodes: int) -> int:
    """Naive partitioner: which of num_nodes owns this key?"""
    return hash_key(key) % num_nodes


# A 4-node cluster places keys across nodes 0..3.
for k in ["user:1000", "user:1001", "user:1002", "user:1003", "user:1004"]:
    print(k, "->", modulo_owner(k, 4))
```

This is correct, it is one machine instruction after the hash, and it spreads keys uniformly across the nodes as long as the hash is good. For a *fixed* cluster it is genuinely fine. The catastrophe is what happens when `num_nodes` changes.

![Naive hash(key) modulo N reshuffles almost every key when the divisor changes from four to five](/imgs/blogs/consistent-hashing-and-data-partitioning-2.webp)

When you add a fifth node, the divisor changes from 4 to 5, and `hash % 4` has essentially no relationship to `hash % 5`. A key whose hash is 1001 maps to node `1001 % 4 = 1` before and `1001 % 5 = 1` after — lucky, unchanged. But a key whose hash is 1002 maps to `1002 % 4 = 2` before and `1002 % 5 = 2`... also unchanged in this cherry-picked case, but across the *whole* keyspace the fraction that stays put is only about `1/N`. Let us not hand-wave it; let us measure it.

```python
import random

random.seed(42)
keys = [f"user:{random.randint(0, 10**9)}" for _ in range(1_000_000)]


def fraction_moved(keys, n_before, n_after) -> float:
    moved = sum(
        1 for k in keys if modulo_owner(k, n_before) != modulo_owner(k, n_after)
    )
    return moved / len(keys)


print(f"4 -> 5 nodes: {fraction_moved(keys, 4, 5):.1%} of keys move")
print(f"8 -> 9 nodes: {fraction_moved(keys, 8, 9):.1%} of keys move")
print(f"100 -> 101 nodes: {fraction_moved(keys, 100, 101):.1%} of keys move")
```

Run it and you get numbers close to these:

```
4 -> 5 nodes: 80.0% of keys move
8 -> 9 nodes: 88.9% of keys move
100 -> 101 nodes: 99.0% of keys move
```

The pattern is exact: going from `N` to `N+1` nodes, the fraction of keys that *stay* is `1/(N+1)` and the fraction that *move* is `N/(N+1)`. At 100 nodes, adding the 101st relocates 99% of all keys. This is the opposite of what you want. The bigger your cluster, the *more* disruptive each membership change becomes — which is precisely backwards, because big clusters change membership often (hardware fails, you autoscale, you do rolling deploys).

Now translate "keys move" into operational consequences. If this is a cache, every moved key is a cache miss on its new node, and the new node fetches from the origin. Add one cache node and you have manufactured a near-total cache flush; the origin database, sized for a 95% hit rate, suddenly takes 20× its normal read load and topples. That is the **cache stampede** the intro described, and it is the exact scenario libketama was written to kill (more on that in section 9). If this is a primary datastore, every moved key is a chunk of data that must be physically streamed to its new owner before the cluster is consistent again, so adding a node triggers a cluster-wide data migration proportional to the *entire* dataset rather than one node's share. Either way, modulo hashing turns the cheapest operational action — adding capacity — into the most expensive.

| Property | `hash(key) % N` | What you actually want |
| --- | --- | --- |
| Lookup cost | O(1), one modulo | O(1) or O(log N), still cheap |
| Load balance (fixed N) | excellent, uniform | excellent |
| Keys moved when N → N+1 | ~N/(N+1), i.e. almost all | ~1/N, only the new node's share |
| Behavior as cluster grows | worse: more keys move each time | better: each change is more localized |
| Cache impact of adding a node | near-total flush, stampede | a small, bounded miss rate |

The last two rows are the entire motivation for consistent hashing. We need a partitioner whose disruption on membership change is proportional to `1/N`, not `N/(N+1)`.

## 3. Consistent hashing: hash the servers too {#consistent-hashing}

**The senior rule: if you want change to be cheap, make ownership a property of *position*, not of *count*.** Karger and colleagues introduced consistent hashing in their 1997 STOC paper, "Consistent Hashing and Random Trees: Distributed Caching Protocols for Relieving Hot Spots on the World Wide Web." The original motivation was not databases at all — it was web caching. A swarm of caching proxies needed to agree on which proxy holds which URL, *without* a central coordinator and *without* re-shuffling everything when a proxy joined or left. Their insight is the one the ring encodes: define a hash function whose output mapping "changes minimally as the range of the function changes," which they achieve by hashing both the items (URLs) and the buckets (proxies) into the same space and assigning each item to the nearest bucket.

The construction is the ring from the mental model. Pick a hash function with a large output range. Hash each node's identifier to get its ring position. Hash each key to get its ring position. A key's owner is the first node clockwise. Here is a minimal but real implementation; it is not toy pseudocode, it runs.

```python
import bisect
import hashlib


def _hash(s: str) -> int:
    """Map any string into the 32-bit ring [0, 2**32)."""
    return int.from_bytes(hashlib.md5(s.encode()).digest()[:4], "big")


class HashRing:
    def __init__(self, nodes=None):
        # ring: sorted list of (position, node) maintained in two parallel arrays
        self._positions = []   # sorted ring positions
        self._nodes = []       # node owning each position, aligned with _positions
        for n in nodes or []:
            self.add(n)

    def add(self, node: str) -> None:
        pos = _hash(node)
        i = bisect.bisect(self._positions, pos)
        self._positions.insert(i, pos)
        self._nodes.insert(i, node)

    def remove(self, node: str) -> None:
        pos = _hash(node)
        i = bisect.bisect_left(self._positions, pos)
        if i < len(self._positions) and self._positions[i] == pos:
            self._positions.pop(i)
            self._nodes.pop(i)

    def owner(self, key: str) -> str:
        if not self._positions:
            raise ValueError("empty ring")
        pos = _hash(key)
        i = bisect.bisect(self._positions, pos)  # first node clockwise
        if i == len(self._positions):
            i = 0  # wrap around the top of the ring
        return self._nodes[i]


ring = HashRing(["node-A", "node-B", "node-C", "node-D", "node-E"])
print(ring.owner("user:42"))   # deterministic, depends only on hashes
print(ring.owner("user:08"))
```

The lookup is a binary search over the sorted ring positions — O(log V) where V is the number of positions on the ring, which is a handful of microseconds even for a large ring. The `owner` method is the clockwise walk made concrete: `bisect` finds the insertion point for the key's hash, which is exactly the first position greater than or equal to it; if we fall off the end of the array, we wrap to index 0, which is the "the top of the ring connects back to the bottom" step.

![A key lookup hashes onto the ring, then walks clockwise to the first node, which becomes its owner](/imgs/blogs/consistent-hashing-and-data-partitioning-3.webp)

The figure above traces one lookup end to end. Key `user:42` hashes to ring position 106 (out of a 256-slot illustrative ring). Walking clockwise, the first node we meet is node C at position 140, so C is the owner. Node B at position 90 is *behind* the key — we already passed it — so it is not the owner, even though it might be numerically closer. Then, if we are replicating (section 5), we keep walking to find replica 2 (node D) and replica 3 (node E). The directional rule matters: ownership is "first node *clockwise*," never "nearest node," and getting that wrong is a classic implementation bug that silently sends a fraction of keys to the wrong place.

Now the payoff. Remove a node and only its arc is orphaned; those keys flow to the next node clockwise, and *no other key moves*. Add a node and it claims a slice of one neighbor's arc; only those keys move. Let us measure the disruption the same way we measured modulo, to make the contrast quantitative.

```python
def fraction_moved_ring(keys, before_nodes, after_nodes) -> float:
    r1 = HashRing(before_nodes)
    r2 = HashRing(after_nodes)
    moved = sum(1 for k in keys if r1.owner(k) != r2.owner(k))
    return moved / len(keys)


base = [f"node-{i}" for i in range(8)]
print(f"8 -> 9 nodes (ring): {fraction_moved_ring(keys, base, base + ['node-8']):.1%}")
# Compare to modulo's 88.9% for the same change.
```

You will see something close to `11%` — and `1/9 ≈ 11.1%` is exactly the fraction the new node *should* take, its fair share. The general result is the headline property of consistent hashing: **adding the (N+1)-th node moves only about `1/(N+1)` of the keys**, the keys that genuinely belong to the new node, instead of modulo's `N/(N+1)`. The disruption shrinks as the cluster grows, which is exactly the right direction. Karger's paper proves this formally: a consistent hash function moves an expected `O(K/N)` items when a bucket is added or removed, where `K` is the number of items and `N` the number of buckets.

There is a catch hiding in that `11%`, though, and it is the reason the *basic* ring is never used as-is in production. Run the experiment with only three or four nodes and you will not get a clean `1/N`; you will get something lumpy. With few random positions on the ring, the arcs between nodes are wildly unequal — one node might own 50% of the ring by sheer luck of where its hash landed, another 5%. The Dynamo paper names this directly: the basic algorithm's "random position assignment of each node on the ring leads to non-uniform data and load distribution," and it is "oblivious to the heterogeneity in the performance of nodes." That lumpiness is the problem virtual nodes solve, and it is the single most important refinement in the whole design.

## 4. Virtual nodes: turning luck into law {#vnodes}

**The rule that makes consistent hashing actually balanced: never give a physical node just one position on the ring. Give it hundreds.** With one position per node and `N` nodes, the arc lengths are determined by `N` random samples on a circle, and the variance is high — the largest arc can easily be several times the smallest. The fix is statistical: instead of placing each physical node once, place it many times, at positions derived from `(node_id, replica_index)`. Each of those positions is a *virtual node* (vnode), and a physical node owns the union of all its vnodes' arcs. With, say, 256 vnodes per physical node, each physical node's total ownership is the average of 256 random arcs, and by the law of large numbers that average converges tightly to the fair share. Lumpy luck becomes smooth law.

![Virtual nodes smooth a lumpy single-token ring into even per-node load by averaging many arcs](/imgs/blogs/consistent-hashing-and-data-partitioning-4.webp)

The before/after above is the point. On the left, one token per node, the arcs are 51%, 12%, 28%, 9% — a 5.7× spread between the hottest and coldest node, a guaranteed hot spot. On the right, with 256 tokens per node, every node converges to roughly 25%, the load is even, and — because each vnode is independent — scaling becomes smooth too. Here is the ring extended with vnodes; the change is small but it is what makes the structure usable.

```python
class VNodeRing:
    def __init__(self, vnodes_per_node: int = 256):
        self.vnodes = vnodes_per_node
        self._positions = []
        self._nodes = []

    def add(self, node: str) -> None:
        for v in range(self.vnodes):
            pos = _hash(f"{node}#{v}")          # one ring point per vnode
            i = bisect.bisect(self._positions, pos)
            self._positions.insert(i, pos)
            self._nodes.insert(i, node)

    def remove(self, node: str) -> None:
        keep = [(p, n) for p, n in zip(self._positions, self._nodes) if n != node]
        self._positions = [p for p, _ in keep]
        self._nodes = [n for _, n in keep]

    def owner(self, key: str) -> str:
        pos = _hash(key)
        i = bisect.bisect(self._positions, pos)
        return self._nodes[i % len(self._positions)]


import collections

ring = VNodeRing(vnodes_per_node=256)
for n in ["A", "B", "C", "D"]:
    ring.add(n)

load = collections.Counter(ring.owner(f"user:{i}") for i in range(1_000_000))
for node in sorted(load):
    print(f"node {node}: {load[node] / 10_000:.1f}% of keys")
```

With 256 vnodes per node and four nodes, the printed shares land within a percent or two of 25% each. Drop `vnodes_per_node` to 1 and re-run, and you will see the lumpy 51/12/28/9-style distribution from the figure. The standard deviation of load shrinks roughly as `1/sqrt(vnodes)`, which is why a few hundred vnodes is usually enough: going from 1 to 100 vnodes cuts the spread by 10×, and from 100 to 256 buys you a bit more polish.

Virtual nodes buy three things at once, and it is worth separating them because real systems tune them independently:

**Even load**, as we just measured — the original motivation. **Smooth, parallel rebalancing**: because a physical node's data is scattered across hundreds of small arcs interleaved with every other node's arcs, when a node fails its arcs are inherited by *many* different successors rather than dumped entirely onto one unlucky neighbor. The recovery streams data *from* many sources *to* many destinations in parallel, which is far faster than one node firehosing its entire dataset to a single replacement. **Heterogeneous capacity**: a node with twice the disk and CPU can simply be assigned twice as many vnodes, so it owns twice the keyspace. The Dynamo paper calls this out explicitly — "the number of virtual nodes that a node is responsible for can be decided based on its capacity, accounting for heterogeneity in the physical infrastructure." That is impossible with one-token-per-node short of awkward weighting hacks.

### Second-order gotcha: more vnodes is not strictly better

Here is the non-obvious part that bites operators. For a long time Cassandra defaulted to `num_tokens: 256`, on the theory that more vnodes means better balance. Then Cassandra 4.0 *reduced* the default to `num_tokens: 16`, and the reasoning is instructive. More vnodes means each node's data is sliced into more, smaller token ranges, scattered across more peers. That improves balance, but it degrades two other things. First, **availability under correlated failure**: with replication factor 3, your data survives losing any 2 nodes only if no single token range happens to have all 3 of its replicas on those 2 nodes plus their successors. The more finely you slice the ring, the more distinct (node-triple) combinations exist, and the higher the probability that *some* range loses all its replicas when a couple of nodes go down together. The TheLastPickle analysis quantified it: on a 6-node RF=3 cluster, single-token placement survives losing any 2 nodes with no data loss, but at `num_tokens: 3` the same 2-node failure makes about 33% of the data unavailable. Second, **streaming and repair cost**: more ranges means more bookkeeping, more SSTables touched, more overhead during repair. Cassandra 4.0 squares this circle by combining a modest 16 tokens with a *replica-aware* token allocation algorithm (`allocate_tokens_for_local_replication_factor: 3`) that places the few vnodes intelligently rather than purely at random, getting good balance without the fine-grained-slicing tax. The lesson: vnode count is a real tradeoff dial between load balance and availability, not a "more is better" knob.

## 5. Replication on the ring: the preference list {#lookup}

**The rule: a key's replicas are not "three random nodes" — they are the next R *distinct* nodes clockwise, and the word "distinct" is doing heavy lifting.** Owning a key on one node is not durable; if that node dies, the key is gone. So distributed databases replicate each key to `R` (or `N`, in Dynamo's notation) nodes. Consistent hashing gives a beautifully simple replica placement: the owner is the first node clockwise, and the replicas are the *next* nodes clockwise after it. Dynamo calls the resulting ordered set of nodes the **preference list** for that key, and every node in the system knows the preference list for every key because it is computable from the ring alone — no directory, no coordination.

![The preference list for a key is its owner plus the next R distinct physical nodes clockwise on the ring](/imgs/blogs/consistent-hashing-and-data-partitioning-5.webp)

The figure shows `R = 3`. Key `u:42` is owned by the first node clockwise (its replica 1), and replicas 2 and 3 are the next two nodes you encounter continuing clockwise. The greyed-out nodes are skipped — one is behind the key, and we will see why the other might be skipped in a moment. Here is the lookup that returns the whole preference list:

```python
class ReplicatedRing(VNodeRing):
    def preference_list(self, key: str, r: int) -> list[str]:
        """Owner plus the next r-1 DISTINCT physical nodes clockwise."""
        pos = _hash(key)
        start = bisect.bisect(self._positions, pos)
        result = []
        n = len(self._positions)
        i = start
        while len(result) < r and len(result) < len(set(self._nodes)):
            node = self._nodes[i % n]
            if node not in result:          # skip vnodes of an already-chosen physical node
                result.append(node)
            i += 1
        return result


ring = ReplicatedRing(vnodes_per_node=128)
for n in ["A", "B", "C", "D", "E"]:
    ring.add(n)
print(ring.preference_list("user:42", r=3))   # e.g. ['C', 'D', 'E']
```

The `if node not in result` check is the "distinct" requirement, and it is exactly the subtlety the Dynamo paper warns about. With virtual nodes, the first `R` positions clockwise from a key might all be vnodes of the *same* physical node, or of fewer than `R` physical nodes — and replicating three copies onto one machine is no replication at all. So the preference list is built by walking clockwise and *skipping positions that map to a physical node already in the list*, until it has `R` distinct physical nodes. Dynamo states it plainly: "the preference list for a key is constructed by skipping positions in the ring to ensure that the list contains only distinct physical nodes." Real systems push this further by making the list *rack-* and *datacenter-aware*, skipping not just repeated nodes but repeated failure domains, so the three replicas land in three different racks or availability zones. That way a rack power failure or an AZ outage cannot take all three copies at once.

The preference list is also where consistent hashing meets [database replication](/blog/software-development/database/database-replication-sync-async-logical-physical). The ring decides *which* nodes hold the replicas; the replication protocol decides *how* writes propagate to them — synchronously for strong consistency, asynchronously for availability, with quorum reads and writes (`R + W > N`) in the Dynamo style to tune the balance. Dynamo also adds the notion of a *coordinator* — typically the first node in the preference list — and *hinted handoff*, where if a replica node is temporarily down, a healthy node further down the list accepts the write with a "hint" and forwards it once the intended node recovers. Hinted handoff is only possible because the preference list extends past the first `R` nodes; the next node clockwise is the natural stand-in.

## 6. Why only K/N keys move: the localized-change property {#minimal-movement}

**The rule worth tattooing on the runbook: in a healthy consistent-hash cluster, a single membership change touches one node's worth of data and no more.** We measured this empirically in section 3; now let us see *why* geometrically, because the intuition is what lets you reason about an incident at 3 a.m.

![Adding a node steals a contiguous arc from its clockwise successor, so only that node's slice of keys moves](/imgs/blogs/consistent-hashing-and-data-partitioning-6.webp)

Consider three nodes owning the arcs 0–120, 120–240, and 240–360 (degrees, for a simple ring). Now add node E at position 60. E carves the arc 60–120 out of node A's territory (A keeps 0–60), and *nothing else changes*: B still owns 120–240, C still owns 240–360, and every key in those arcs stays exactly where it was. The only keys that move are the ones in 60–120, and they move from A to E. That is one arc, roughly `1/N` of the ring, exactly the new node's fair share. Removal is the mirror image: delete a node and its arc is absorbed by the next node clockwise, again touching only one arc's worth of keys.

With virtual nodes the same property holds, just spread out: a new physical node's 256 vnodes each carve a small arc out of whichever node currently owns that spot, so the `1/N` of moved keys is sourced from *many* nodes in small pieces rather than from one neighbor in a big chunk. This is what makes vnode rebalancing fast and parallel — the work fans out — but the *total* data moved is still just the new node's share. Here is the audit you can run to convince yourself, and that you can adapt into a real "how much will this rebalance move?" pre-flight check before an operation:

```python
def rebalance_cost(before_nodes, after_nodes, sample_keys, vnodes=256):
    r1 = VNodeRing(vnodes); [r1.add(n) for n in before_nodes]
    r2 = VNodeRing(vnodes); [r2.add(n) for n in after_nodes]
    moves = collections.Counter()
    for k in sample_keys:
        a, b = r1.owner(k), r2.owner(k)
        if a != b:
            moves[(a, b)] += 1
    total = sum(moves.values())
    print(f"{total / len(sample_keys):.1%} of keys move; sources/sinks:")
    for (src, dst), c in sorted(moves.items(), key=lambda x: -x[1])[:6]:
        print(f"  {src} -> {dst}: {c / len(sample_keys):.2%}")


sample = [f"user:{i}" for i in range(200_000)]
rebalance_cost(["A", "B", "C", "D"], ["A", "B", "C", "D", "E"], sample)
```

You will see roughly `20%` of keys move (the fair share of the new fifth node, `1/5`), and — this is the vnode magic — the moves are sourced *from all four existing nodes roughly equally* and sink *into E*. No single existing node loses more than its proportional contribution. Contrast that with single-token consistent hashing, where the new node would steal its entire arc from *one* neighbor, hammering that one node's disk and network during the migration while the others sat idle. The cost-audit pattern above is the difference between a planned, predictable rebalance and a surprise.

> The deepest property of the ring is not that it balances load — vnodes do that. It is that it makes the *cost of change* proportional to the *size of the change*, not the size of the system. Adding one node costs one node's worth of data movement, whether your cluster has 5 nodes or 500.

## 7. Hash versus range partitioning, revisited {#hash-vs-range}

**The rule: choose hash partitioning when your access pattern is point lookups, range partitioning when it is scans — and never pretend one scheme is strictly better.** Consistent hashing is a hash-partitioning scheme, and hashing's great strength is also its great weakness. By running the key through a hash before placing it, you scatter even adjacent keys to opposite ends of the ring. That is wonderful for load: a workload that writes monotonically increasing keys — timestamps, auto-increment IDs, UUIDv1 — would create a brutal hot spot under range partitioning (every write lands on the node owning the highest range), but under hashing those sequential keys diffuse uniformly. It is the same diffusion that makes [random UUIDs hurt B-tree locality](/blog/software-development/database/random-uuids-are-killing-your-database-performance) actually *help* at the partitioning layer: random spread is poison for a single-node ordered index but ideal for spreading load across shards.

![Hash partitioning spreads load but kills range scans; range partitioning supports scans but invites hotspots on sequential keys](/imgs/blogs/consistent-hashing-and-data-partitioning-7.webp)

The matrix above lays out the five properties that actually decide the choice. The brutal one is the **range scan** row. Under hashing, the keys `c`, `d`, `e`, `f` are scattered across every partition, so a query for "all keys between c and f" has no choice but to fan out to *every* node and gather the results — a scatter-gather that gets slower as the cluster grows. Under range partitioning, that same query touches exactly one partition. This is why Kleppmann frames it as a genuine dichotomy in DDIA Chapter 6 rather than a clear winner: hash partitioning by key "destroys the ordering of keys, making range queries inefficient," while range partitioning "keeps keys in sorted order... but the risk is that of hot spots."

Real systems split along exactly this line. **Dynamo and Cassandra** hash by key (Cassandra calls the hash output a *token* and the ring a *token ring*) because their target workload is high-volume point reads and writes where even load is paramount. **HBase, Bigtable, and Spanner** use range partitioning (they call the ranges *tablets* or *splits*) because their target workload includes large ordered scans — time-series, analytics, prefix queries — where touching one partition per scan is the whole point. MongoDB lets you choose: *ranged* sharding for scan-heavy collections, *hashed* sharding for write-heavy ones with monotonic shard keys.

There is a hybrid worth knowing, because it shows up constantly in practice and Kleppmann highlights it: a **compound key** where the first component is hashed (or chosen for spread) and the rest is kept ordered. Cassandra's `PRIMARY KEY ((user_id), timestamp)` hashes `user_id` to pick the partition but stores each user's rows sorted by `timestamp` *within* that partition. You lose cross-user range scans but keep per-user ones, and you still spread load across users. That pattern — hash the high-cardinality dimension to spread, keep the scan dimension ordered locally — is the most common way real schemas get the best of both worlds.

| Need | Reach for hashing | Reach for ranges |
| --- | --- | --- |
| Point lookups by exact key | yes, ideal | fine |
| Range scans / prefix queries | no, scatter-gather | yes, ideal |
| Monotonically increasing keys | yes, diffuses the hotspot | no, creates a hotspot |
| Even load with skewed key popularity | partial — see section 8 | poor |
| Per-entity ordered access | compound key (hash + sorted) | yes |

## 8. Hot spots: when even a perfect ring is not enough {#hotspots}

**The rule: consistent hashing balances *keys*, not *traffic* — and a single popular key can still melt one node no matter how good your ring is.** Virtual nodes guarantee that each node owns roughly the same *number* of keys. They guarantee nothing about how often those keys are *accessed*. If one key — a celebrity's profile, a viral post, a global config object — receives a million reads a second, that key lives on exactly one owner (plus its replicas), and that owner is on fire while every other node naps. The ring did its job perfectly and you still have a hot spot. This is the "hot key" problem, and it is the limit of what placement alone can do.

The mitigations layer on top of consistent hashing rather than replacing it:

**Key salting / fan-out.** For a known hot key, append a small random suffix to spread it across `K` artificial keys (`celebrity:profile:0` ... `celebrity:profile:9`), each landing on a different ring position and thus a different node. Reads pick a random suffix; writes must update all `K`. This trades write amplification and read complexity for read-throughput, and it only works when you can *predict* which keys are hot. Kleppmann describes exactly this in DDIA — adding "a random number to the beginning or end of the key" to split a hot key across partitions — and notes the bookkeeping cost: any read must now query all the variants.

**Read replicas and caching.** Because the preference list already places `R` replicas, you can serve reads from any of them, multiplying read throughput by `R` for free. For extreme cases, front the hot key with a dedicated cache; this is one of the canonical jobs of [Redis in production](/blog/software-development/database/redis-applications-and-optimization), absorbing the read storm so the datastore behind it never sees it.

**Good key design.** The cheapest fix is upstream: design keys so popularity is naturally spread. Bucketing a global counter into per-shard counters that you sum on read, or partitioning a busy entity by a natural secondary dimension, prevents the hot key from forming in the first place. As with most distributed-systems problems, the best hot-spot fix is the one that makes the hot spot impossible rather than the one that survives it.

The honest summary is that consistent hashing solves *spatial* skew (uneven key counts per node) and is largely powerless against *temporal/popularity* skew (uneven request counts per key). Knowing which kind of skew you have tells you whether to reach for vnodes or for salting.

## 9. Bounded-load consistent hashing: capping any node's share {#bounded-load}

**The rule: classic consistent hashing bounds *expected* load but not *worst-case* load — bounded-load hashing fixes that by letting a key spill to the next node when its first choice is full.** Even with vnodes, the load on a node is a random variable, and with dynamic key popularity the variance can be uncomfortable. In 2016 Mikkel Thorup and colleagues at Google published "Consistent Hashing with Bounded Loads," which adds a hard guarantee: no node ever exceeds `(1 + ε)` times the average load, for a tunable `ε > 0`.

The mechanism is a small, elegant extension of the clockwise walk. Compute each node's capacity as `(1 + ε) × average_load` (floored or ceiled to an integer). To place a key, walk clockwise as usual — but if the first node is already at capacity, *skip it and keep walking* to the next node with spare room. The Google research blog states the rule directly: when a bin reaches capacity, clients "move clockwise until they find the first bin with spare capacity." The remarkable theoretical result is the movement bound: every insertion or deletion causes only `O(1/ε²)` other keys to relocate, *independent of the total number of keys or nodes*. You pay a little consistency (some keys no longer live on their "natural" owner because it was full) for a hard ceiling on imbalance, and the price in churn is constant.

```python
def bounded_owner(key, ring: VNodeRing, loads: dict, capacity: int) -> str:
    """Place key on the first clockwise node with spare capacity (<= capacity)."""
    pos = _hash(key)
    start = bisect.bisect(ring._positions, pos)
    n = len(ring._positions)
    for step in range(n):
        node = ring._nodes[(start + step) % n]
        if loads.get(node, 0) < capacity:
            loads[node] = loads.get(node, 0) + 1
            return node
    raise RuntimeError("cluster at capacity")  # sum of capacities exceeded total keys


# With eps=0.25, capacity = ceil(1.25 * avg_load); no node exceeds 125% of average.
```

This is not academic. The blog reports two production deployments: Google Cloud Pub/Sub adopted it and saw "substantial improvement on uniformity of the load allocation," and — the number people quote — Vimeo implemented it in HAProxy for their video CDN and "decreased cache bandwidth by a factor of almost 8," because keeping each cache node's load bounded dramatically improved cache locality and eliminated the thundering-herd reshuffles that an unbounded scheme produced. Bounded-load hashing is the right tool when you have a hard SLO on per-node load — a CDN, a stateful load balancer, a sharded cache — and can tolerate a key occasionally living one hop from its natural home. The smaller you set `ε`, the tighter the load cap but the more keys spill and the more churn you accept; `ε` is the single dial that trades uniformity against consistency.

## 10. Rendezvous (HRW) hashing: no ring, just a max {#alternatives}

**The rule: when your node set is small and you want even better balance than a ring with no vnode bookkeeping, rendezvous hashing is often the simpler, better choice.** Rendezvous hashing — also called Highest Random Weight (HRW) — predates the popularity of ring-based consistent hashing; Thaler and Ravishankar published it at the University of Michigan in 1996, a year before Karger's STOC paper. It answers the same question, "which node owns this key," with a completely different mechanism: instead of placing nodes and keys on a shared ring, you hash the *pair* `(key, node)` for every node and assign the key to the node with the highest hash.

```python
def hrw_owner(key: str, nodes: list[str]) -> str:
    """Highest Random Weight: the node whose (key, node) hash is largest owns the key."""
    return max(nodes, key=lambda node: _hash(f"{key}:{node}"))


def hrw_preference_list(key: str, nodes: list[str], r: int) -> list[str]:
    """Top-r nodes by weight = the natural replica set, no skipping needed."""
    return sorted(nodes, key=lambda node: _hash(f"{key}:{node}"), reverse=True)[:r]
```

That is the whole algorithm — there is no ring to maintain, no vnodes, no sorted array. Its properties are excellent. The load is *naturally* uniform without vnodes, because each key independently picks a random winner among the nodes, so each node wins about `1/N` of the keys with low variance. Membership change is minimal in exactly the right way: adding a node only steals the keys for which the new node's `(key, node)` hash happens to be the new maximum (about `1/(N+1)` of keys, the fair share), and removing a node only re-assigns *that* node's keys, each to its second-highest-weight node — every other key keeps its winner. The preference list comes for free: the top-`R` nodes by weight, already distinct, no skip logic. And weighting for heterogeneous capacity is a clean formula on the hash rather than a vnode count.

The catch is the cost model. A naive HRW lookup is `O(N)` — you hash the key against *every* node and take the max — versus the ring's `O(log V)`. For a few dozen nodes that is nothing (a few dozen hashes is faster than you can measure), and HRW is the better choice. For thousands of nodes it becomes the bottleneck, which is why ring-based schemes dominate at hyperscale. (There are `O(log N)` HRW variants using skeleton trees, but they reintroduce the bookkeeping HRW was meant to avoid.) The practical rule: **HRW for small-to-medium node counts where its perfect balance and zero bookkeeping win; the ring for very large clusters where `O(log V)` lookup matters.** HRW shows up in CRUSH (Ceph's data placement), in some Kafka-style partition assigners, and in load balancers where the node count is modest.

## 11. Jump consistent hash: zero memory, five lines, one limitation {#jump-hash}

**The rule: if your "nodes" are just numbered buckets 0..N−1 and you only ever grow or shrink at the end, jump consistent hash is unbeatable — and almost useless otherwise.** Lamping and Veach at Google published "A Fast, Minimal Memory, Consistent Hash Algorithm" in 2014. It maps a 64-bit key and a bucket count `N` to a bucket in `[0, N)` using *no data structure at all* — no ring, no node list, no memory — in a tight loop that runs in `O(ln N)` expected time. The canonical implementation is famously about five lines:

```python
def jump_consistent_hash(key: int, num_buckets: int) -> int:
    """Lamping & Veach 2014: map a 64-bit key to a bucket in [0, num_buckets)."""
    b, j = -1, 0
    while j < num_buckets:
        b = j
        key = (key * 2862933555777941757 + 1) & 0xFFFFFFFFFFFFFFFF  # 64-bit LCG step
        j = int((b + 1) * (float(1 << 31) / float((key >> 33) + 1)))
    return b
```

The intuition behind the loop is a probability argument. Picture the buckets being added one at a time. When the bucket count grows from `b+1` to the next jump target `j`, a key should "jump" to a new bucket only with probability `(b+1)/j` — exactly the fraction needed so that each bucket ends up with `1/N` of the keys and so that growing `N` perturbs the assignment minimally. The loop uses the key as a seed for a pseudo-random sequence (a 64-bit linear congruential generator) to decide each jump deterministically, computing the *next* bucket the key would jump to in closed form rather than iterating one bucket at a time. The expected number of jumps is `O(ln N)`, and because there is no state, lookups are blisteringly fast and trivially parallel.

The Lamping-Veach paper compares it head-to-head with ring-based consistent hashing and wins on every quantitative axis: it requires no storage (the ring needs memory proportional to nodes × vnodes), it is faster per lookup, and it divides keys more evenly across buckets — the paper reports a key-distribution standard deviation far below the ring's, because it is not subject to the random-arc-length lumpiness that forced vnodes onto the ring in the first place.

So why is the ring still everywhere? The one fatal limitation: **buckets must be numbered `0..N−1`, and you can only add or remove the *last* bucket.** Jump hash has no concept of node identity — bucket 5 is just "the sixth bucket," not "the node at 10.0.0.5." If you remove bucket 2 from the middle, jump hash cannot express that; it can only shrink from `N` to `N−1` by dropping bucket `N−1`. That makes it perfect for **data storage where you control the bucket numbering and scale by appending shards** — sharded storage, partitioned files, consistent assignment of work to a numbered worker pool — and unsuitable for **distributed web caching or any cluster where arbitrary, identity-bearing nodes come and go** (the exact use case Karger's ring was built for). The paper itself draws this line: jump hash is "more suitable for data storage applications than for distributed web caching." It is the sharpest tool in the box for the narrow job it fits.

## 12. The family tree: choosing a scheme {#scheme-tree}

![The family tree of key-to-node mapping schemes: modulo, ring, rendezvous, and jump occupy distinct tradeoff points](/imgs/blogs/consistent-hashing-and-data-partitioning-9.webp)

The tree above organizes everything we have covered as answers to one question — "which node owns this key, and what moves when the membership changes?" Naive `hash % N` remaps almost everything on resize and is disqualified for any cluster that ever changes size. The ring (with vnodes and a preference list, optionally with a bounded-load cap) is the general-purpose workhorse: identity-bearing nodes, arbitrary join/leave, good balance, `O(log V)` lookup. Rendezvous/HRW gives perfect balance with zero bookkeeping but `O(N)` lookup, ideal for modest node counts. Jump hash gives zero memory and the best balance of all but only for sequentially numbered buckets that grow at the end. Here is the decision compressed into a table you can keep:

| Scheme | Lookup cost | Memory | Balance | Arbitrary node id? | Best for |
| --- | --- | --- | --- | --- | --- |
| `hash % N` | O(1) | none | great (fixed N) | n/a | a cluster that never resizes |
| Ring + vnodes | O(log V) | O(nodes × vnodes) | good (tunable) | yes | general distributed DB / cache |
| Bounded-load ring | O(log V) | O(nodes × vnodes) | hard-capped | yes | per-node SLO, CDN, LB |
| Rendezvous (HRW) | O(N) | none | excellent | yes | small/medium clusters, CRUSH |
| Jump hash | O(ln N) | none | best | no (0..N−1 only) | numbered shards, worker pools |

## 13. Rebalancing strategies: how the cluster actually grows {#rebalancing}

**The rule, straight from DDIA: the way you split the keyspace into partitions is a separate decision from how you place those partitions on nodes, and conflating them is the source of most rebalancing pain.** Kleppmann's Chapter 6 makes a point that trips up almost everyone learning this material: the "consistent hashing" ring is *one* way to decide partition boundaries, but it is not the only one, and several widely deployed systems deliberately avoid it. He lays out three rebalancing strategies, and they map cleanly onto how real databases grow.

![Three rebalancing strategies: fixed partition count, dynamic splitting, and proportional-to-nodes each move different amounts of data](/imgs/blogs/consistent-hashing-and-data-partitioning-8.webp)

**Fixed number of partitions.** Create *far more* partitions than nodes up front — say 1000 partitions for a 10-node cluster, 100 per node — and never change the partition count. When you add a node, it simply *steals whole partitions* from existing nodes until everything is balanced again; when you remove one, its partitions are redistributed. The partition-to-node assignment changes; the key-to-partition mapping never does. This is what **Riak, Elasticsearch, Couchbase, and Voldemort** do. Riak's ring is exactly this: a 160-bit space divided into a fixed `ring_creation_size` (default 64) of equal partitions, each managed by a *vnode* process, and the number of vnodes per physical node is just `ring_size / num_nodes`. Kleppmann notes the key advantage — moving whole partitions is operationally simple and the partition boundaries are stable — and the key constraint: you must choose the partition count up front to be high enough for your *maximum* future cluster size, because you can never change it. Pick too few and you cannot scale out far; pick too many and each partition is tiny and the per-partition overhead dominates. This is the strategy where Kleppmann's "the term consistent hashing is confusing" caveat lands hardest: Riak's fixed-partition ring is *not* Karger's consistent hashing even though both use a ring, and conflating them leads to wrong intuitions about what moves on a rebalance.

**Dynamic partitioning.** Start with one (or a few) partitions and *split* a partition in two when it grows past a size threshold, *merging* adjacent small partitions when they shrink — exactly like a B-tree splits and merges pages. The partition count tracks the data volume, so a small dataset has few partitions and a large one has many, with no up-front guess required. This is what **HBase and MongoDB (ranged sharding)** do, and it is the natural fit for *range* partitioning because splits respect key order. The cost is that a newly-created cluster has only one partition on one node until the first split, so you often *pre-split* to get parallelism from the start. Kleppmann frames the win as adaptivity — partition count follows data size — and the gotcha as the cold-start single-partition bottleneck.

**Partitioning proportionally to nodes.** Make the number of partitions proportional to the number of *nodes* — a fixed number of partitions *per node*, not per dataset. When you add a node, it picks some number of existing partitions to split, taking half of each split for itself. This keeps each partition roughly the same size as the dataset grows (because adding data means adding nodes means adding partitions) and is exactly the **Cassandra and ScyllaDB vnode** model: a fixed `num_tokens` per node means total partitions scale with node count. Kleppmann notes this is the strategy most directly tied to consistent hashing's random-boundary placement, with the caveat that random split points can produce uneven splits — which is precisely why Cassandra 4.0 added the replica-aware token allocator we discussed in section 4.

| Strategy | Partition count | When you add a node | Used by |
| --- | --- | --- | --- |
| Fixed count | constant, >> nodes, chosen up front | move whole partitions to it | Riak, Elasticsearch, Couchbase |
| Dynamic split | grows with data volume | reassign partitions; split at threshold | HBase, MongoDB (ranged), Bigtable |
| Proportional to nodes | fixed per node, grows with cluster | split existing partitions, take half | Cassandra, ScyllaDB (vnodes) |

### Request routing: finding the partition after you have placed it

Placing data is half the problem; *finding* it on a read is the other half, and Kleppmann treats it as its own topic ("request routing"). Once a key's partition can move between nodes, a client needs to know which node to talk to. There are three architectures. **Route through any node** (gossip-based): send the request to any node, which forwards it to the right one — Cassandra and Riak gossip the ring/token map between all nodes so any node can route. **A routing tier**: a dedicated layer (like MongoDB's `mongos` or a partition-aware proxy) holds the map and forwards. **A coordination service**: nodes and clients subscribe to a service like ZooKeeper that holds the authoritative partition-to-node map (HBase, Kafka, SolrCloud historically). The consistent-hash ring makes the first option cheap because the map *is* the ring and the ring is small and gossipable; that is a quiet but real advantage of hashing over schemes that need a big explicit partition directory.

## 14. How the real systems wire it together

It is worth seeing the pieces assembled in named production systems, because each makes different choices and the differences are instructive.

**Amazon Dynamo** (the 2007 paper, the intellectual source for most of this) uses ring-based consistent hashing with virtual nodes for partitioning, the preference list of `N` distinct nodes for replication, quorum reads and writes (`R + W > N`) for tunable consistency, hinted handoff for temporary failures, and Merkle-tree anti-entropy for permanent ones. Its descendant **DynamoDB** (the managed service) hides the ring entirely behind an API but uses the same partitioning ideas, auto-splitting partitions as a table's throughput or size grows — closer to dynamic partitioning than the original's fixed ring.

**Apache Cassandra and ScyllaDB** use the proportional-to-nodes vnode model: each node owns `num_tokens` positions on the token ring (256 historically, 16 in Cassandra 4.0 with smart allocation), data is partitioned by the hash (token) of the partition key, and replication walks the ring for `RF` distinct nodes with rack/DC awareness. ScyllaDB is a C++ rewrite of Cassandra's model with a shard-per-core architecture layered on top, so it partitions twice: across nodes by token, then across cores within a node.

**Riak** uses the fixed-partition ring: a 160-bit space split into `ring_creation_size` equal partitions (default 64), each a vnode, replicated to `N` (default 3) consecutive partitions, with nodes claiming partitions to balance ownership. Adding a node triggers a *claim* algorithm that hands it whole partitions.

**Memcached clients** (via libketama) and **Redis Cluster** sit at the cache end. Libketama is pure client-side consistent hashing — the servers are dumb, the client holds the ring and picks the server — and it was written specifically to stop the add-a-cache-node stampede. Redis Cluster takes the fixed-partition approach with a twist: it defines exactly **16384 hash slots** (`CRC16(key) % 16384`), assigns slots to nodes, and moves slots between nodes to rebalance — a fixed partition count chosen as a nice round number that is plenty for any realistic Redis cluster while keeping the slot map small enough to gossip.

The throughline: every one of these is the same `owner = f(key, members)` function with the same goal — keep load balanced and keep the cost of changing `members` proportional to the change. They differ only in *which* implementation of `f` and *which* rebalancing strategy fits their workload.

## Case studies from production

### 1. The cache node that flushed the cache

A team running a 12-node Memcached fleet with a client library that defaulted to `hash(key) % N` added two nodes during a traffic spike to add headroom. The instant the new nodes registered, the cache hit rate fell from 96% to about 14% — almost exactly the `2/14` you would predict for going from 12 to 14 nodes under modulo. The origin MySQL fleet, sized for a 96% hit rate, took roughly 20× its normal read load and the read replicas fell over, which looked like a *database* incident and sent everyone debugging the wrong system. The actual root cause was the client's hash policy. The fix was a one-line change to the client config to use ketama (consistent hashing); the *real* fix was a policy that no cache client may use modulo distribution, enforced in the shared library. The lesson is that modulo hashing fails *silently* until a membership change, so it survives code review and load tests and only detonates in production at the worst time.

### 2. The single-token hot node

A small Cassandra cluster — five nodes, single-token (vnodes disabled, as some early operators did for "predictability") — developed one node running at 90% disk while the others sat at 30%. The token positions, assigned manually and never rebalanced after a node replacement, left one node owning a 40% arc of the ring. Every query touching that arc piled onto one node. The first hypothesis was a hardware problem (slow disk on that host); the second was a hot partition. The actual cause was ring imbalance from manual token assignment. Switching to vnodes (`num_tokens`) and running `nodetool cleanup` and repair rebalanced ownership to ~20% each within a day. Lesson: single-token rings need careful, *ongoing* token math; vnodes exist precisely so you do not have to do that math by hand.

### 3. The viral key that vnodes could not save

A social app on a well-tuned vnode Cassandra cluster — perfectly balanced key counts across 30 nodes — saw three nodes spike to 100% CPU while the other 27 idled, during a single celebrity event. The on-call assumed ring imbalance and ran a rebalance, which did nothing because the ring *was* balanced. The real cause was a single hot key (the celebrity's feed object) plus its two replicas: three nodes, one viral key, a million reads a second. Consistent hashing balances key *count*, not key *traffic*. The fix was application-level: salt the hot key into 16 variants spread across the ring, and front it with a Redis cache layer to absorb the read storm. Lesson: when load is skewed but the ring is balanced, the skew is *temporal* (popularity), and placement cannot fix it — you need salting, replicas, or caching.

### 4. Too many vnodes, lost data

An operator running Cassandra at `num_tokens: 256` for "best balance" lost two nodes in the same rack to a power event on an RF=3 cluster and discovered a slice of data was unavailable — even though "any 2 nodes can fail" was the assumed guarantee. With 256 tokens per node, the number of distinct replica-set combinations was so high that *some* token range had all three replicas within the two failed nodes' replica span. The fix was to migrate to `num_tokens: 16` with `allocate_tokens_for_local_replication_factor: 3` and rack-aware placement, trading a little balance for a large availability gain. Lesson: vnode count is an availability dial, not just a balance dial; the Cassandra 4.0 default change to 16 encodes exactly this hard-won lesson.

### 5. Jump hash on identity-bearing nodes

A team building a sharded blob store chose jump consistent hash for its zero memory and excellent balance, mapping `hash(blob_id)` to a shard in `[0, N)`. It worked beautifully until a shard in the *middle* of the range was decommissioned for hardware retirement. Jump hash cannot remove a middle bucket — it only shrinks at the end — so removing shard 7 of 20 was inexpressible, and a naive "renumber the shards" remap relocated a huge fraction of blobs. The fix was to introduce a stable indirection: jump-hash to a *logical* shard number, then a small, rarely-changing lookup table mapped logical shards to physical hosts, so decommissioning a host meant editing one table row, not renumbering. Lesson: jump hash's constraint (sequential, end-only buckets) is non-negotiable; if your "nodes" have identity and can leave from the middle, either add an indirection layer or use a ring/HRW instead.

### 6. The unbalanced rebalance that saturated one link

A Riak cluster (fixed-partition ring) had a node replaced after a disk failure. The replacement bootstrapped by streaming all of the failed node's partitions, but because of how the partitions were claimed, nearly all of them came from a *single* surviving neighbor, saturating that one node's network link and slowing every query routed through it for hours. The expectation had been that recovery would draw from many nodes in parallel. The cause was a claim/handoff configuration that concentrated source partitions. The fix tuned the transfer concurrency and the claim algorithm to spread sources, and the broader fix was to verify, *before* an operation, where the data would move from — the `rebalance_cost` audit pattern from section 6. Lesson: "only K/N keys move" is true of the *total*, but *where they move from* depends on vnode/partition placement; a migration can still hot-spot one source link if placement is concentrated.

### 7. Bounded loads rescue a stateful load balancer

A video platform routed user sessions to backend cache servers with plain consistent hashing for cache affinity. A few popular streams produced cache nodes at 3–4× the average load while others idled, blowing the per-node bandwidth SLO and forcing over-provisioning. Switching the HAProxy balancer to consistent hashing *with bounded loads* (`ε ≈ 0.25`, capping each node at 125% of average) let overloaded nodes spill sessions to the next node clockwise. Per-node bandwidth flattened and cache bandwidth dropped sharply — the same class of win Google reported for Vimeo's "factor of almost 8." Lesson: when you have a hard per-node SLO and can tolerate a key occasionally living one hop from home, bounded-load hashing turns a soft expected-load guarantee into a hard worst-case one for a constant churn cost.

### 8. The scatter-gather that scaled the wrong way

An analytics service hash-partitioned event data by event ID for even write load, which worked great for writes. Then product asked for "show me all events between two timestamps," a range scan. Under hash partitioning every timestamp range is scattered across all partitions, so each scan fanned out to all 40 nodes and gathered results — and got *slower* as the cluster grew, the opposite of scaling. The first instinct was to add nodes, which made it worse. The real fix was a schema change to a compound key: hash a coarse time bucket to spread load, keep fine timestamps ordered *within* each bucket, so a range scan touched only the few partitions for the relevant buckets. Lesson: hash partitioning and range scans are fundamentally opposed (section 7); if you need scans, design for them in the key, because no amount of nodes fixes a scatter-gather.

### 9. Rendezvous beats the ring at small scale

A team maintaining a ring with vnodes for a 6-node service spent real effort tuning vnode counts to keep balance acceptable, and still saw a few percent imbalance plus the operational weight of the vnode bookkeeping. They replaced it with rendezvous (HRW) hashing — `max` over `hash(key, node)` for six nodes, six hashes per lookup, no ring to maintain. Balance improved to near-perfect, the preference list became "top-3 by weight" with no skip logic, and the code shrank. The `O(N)` lookup cost was irrelevant at six nodes (six hashes is sub-microsecond). Lesson: the ring's `O(log V)` lookup is a hyperscale optimization; at small-to-medium node counts, rendezvous hashing is often simpler *and* better balanced, and reaching for a ring by reflex is over-engineering.

### 10. The compound-key hotspot nobody predicted

A multi-tenant SaaS hash-partitioned by `tenant_id` for isolation and spread, which balanced *tenants* evenly across nodes. Then one enterprise customer onboarded with 100× the data and traffic of any other tenant, and that single tenant's partition saturated its node. The ring was balanced by tenant *count*, but tenants were not equal in size. The fix was to sub-partition the whale tenant by a secondary key (`(tenant_id, region)` for large tenants, plain `tenant_id` for the rest), spreading the whale across multiple ring positions. Lesson: hash partitioning balances the *number* of partition-key values, not their *weight*; whenever the per-key data or traffic is itself skewed, you need a finer partition key or per-key sub-partitioning, which is the spatial-skew cousin of case study 3's temporal skew.

## When to reach for consistent hashing, and when not to

Reach for ring-based consistent hashing with virtual nodes when:

- You have a **stateful, distributed** system — a database, a cache, a sharded store — where keys must map to nodes and the **node set changes** (failures, autoscaling, deploys), and you cannot afford a global reshuffle on each change.
- You need **point lookups** computed locally from key and membership with **no central directory**, and the cluster is **large enough** (dozens to thousands of nodes) that `O(log V)` lookup and gossipable membership matter.
- You want **replication placement for free** (the preference list) and **heterogeneous capacity** support (more vnodes for bigger nodes).
- You can layer hot-key mitigations (salting, replicas, caching) on top, because you understand the ring balances key *count*, not key *traffic*.

Reach for a **variant** when:

- **Bounded-load** if you have a hard per-node SLO (CDN, stateful load balancer) and can tolerate keys spilling one hop when their owner is full.
- **Rendezvous / HRW** if your node count is **small to medium**, you want perfect balance with zero bookkeeping, and `O(N)` lookup is negligible.
- **Jump hash** if your buckets are **sequentially numbered** (`0..N−1`), you **only grow or shrink at the end**, and you want zero memory and the best balance — a numbered worker pool or appended shards.

Skip consistent hashing (or hashing entirely) when:

- Your workload is dominated by **range scans or ordered access** — reach for **range partitioning** (HBase/Bigtable/Spanner style) and accept the sequential-key hotspot risk, or use a **compound key** that hashes the spread dimension and keeps the scan dimension ordered.
- Your cluster is **fixed-size and never resizes** — plain `hash % N` is simpler and there is no reshuffle to avoid.
- You actually need a **partition directory anyway** (heavy operational tooling around explicit partition placement) — a fixed-partition scheme (Riak/Elasticsearch) with an external map may be operationally simpler than a pure ring, which is exactly Kleppmann's point that the ring is one option among several, not a default.
- The hard problem is **temporal/popularity skew**, not spatial placement — no partitioner fixes a single viral key; that is a caching and key-design problem.

The deepest takeaway is the one the ring diagram encodes: the entire field is the search for an `owner = f(key, members)` whose disruption on a membership change is proportional to the *size of that change*, not the size of the system. Modulo fails because it globalizes change; the ring, rendezvous, and jump hash each succeed by localizing it in a different way. Pick the one whose constraints — identity-bearing nodes, lookup cost, scan support, per-node SLO — match your workload, and you get a system that grows by adding capacity instead of breaking when you do.

## Further reading

- Karger et al., ["Consistent Hashing and Random Trees"](https://www.akamai.com/site/en/documents/research-paper/consistent-hashing-and-random-trees-distributed-caching-protocols-for-relieving-hot-spots-on-the-world-wide-web-technical-publication.pdf) (STOC 1997) — the origin, and the source of the term.
- DeCandia et al., ["Dynamo: Amazon's Highly Available Key-value Store"](https://www.allthingsdistributed.com/files/amazon-dynamo-sosp2007.pdf) (SOSP 2007) — virtual nodes, preference lists, quorums, hinted handoff.
- Martin Kleppmann, *Designing Data-Intensive Applications*, Chapter 6 (Partitioning) — the definitive treatment of partitioning strategies, request routing, and the "consistent hashing is confusingly named" caveat.
- Mirrokni, Thorup, Zadimoghaddam, ["Consistent Hashing with Bounded Loads"](https://research.google/blog/consistent-hashing-with-bounded-loads/) (Google Research, 2017) — the `(1+ε)` cap and the Vimeo/HAProxy result.
- Lamping & Veach, ["A Fast, Minimal Memory, Consistent Hash Algorithm"](https://arxiv.org/abs/1406.2294) (2014) — jump consistent hash.
- Thaler & Ravishankar, ["Using Name-Based Mappings to Increase Hit Rates"](https://en.wikipedia.org/wiki/Rendezvous_hashing) — rendezvous / HRW hashing.
- [Apache Cassandra: Dynamo and data distribution](https://cassandra.apache.org/doc/latest/cassandra/architecture/dynamo.html) and [Riak: Vnodes & the ring](https://docs.riak.com/riak/kv/latest/learn/concepts/vnodes/index.html) — production docs.
- Richard Jones, ["libketama: a consistent hashing algo for memcache clients"](https://www.metabrew.com/article/libketama-consistent-hashing-algo-memcached-clients) — the client-side cache story.
- Sibling posts on this blog: [partitioning and sharding](/blog/software-development/database/database-partitioning-and-sharding), [database replication](/blog/software-development/database/database-replication-sync-async-logical-physical), [random UUIDs are killing your database performance](/blog/software-development/database/random-uuids-are-killing-your-database-performance), and [Redis in production](/blog/software-development/database/redis-applications-and-optimization).
