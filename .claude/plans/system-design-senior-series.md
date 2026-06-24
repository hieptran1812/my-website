# System Design, Like a Senior — series roadmap

**Folder:** `content/blog/software-development/system-design/`
**Category:** `software-development` · **Subcategory:** `System Design`
**Depth:** deep-dive (≥11,000 words, ≥8 figures each) · **Language:** English (verify gate is English-only)
**Voice:** staff/principal engineer — intuition before math, tradeoffs not definitions, real production references, problem-solving framework throughout.
**Every post carries (cross-cutting, mandatory):**
- **Trade-off analysis** — an explicit decision matrix / "why not the alternative" section: what you gain, what you pay, when each option wins. No recommendation without the cost named.
- **Optimization** — where the bottleneck actually is, how to make it faster/cheaper/more reliable, and how to measure the win (latency/throughput/cost/$ numbers).
- **Problem-solving framing** — pose the real engineering problem, reason to the design, stress-test it.
**Already in folder (NOT part of series, avoid duplicating):** design-patterns-guide, livekit-real-time-communication, rabbitmq-production-architecture-scaling.
**Cross-link into existing series:** database (40 posts), message-queue (40 posts), site-reliability-engineering.

**Execution:** parallel-agent waves; each wave = prose agents author markdown + diagram DSL, main session renders figures, runs `verify-post.sh` (deep-dive), then commits ONLY that wave's md+webp (explicit paths, never `git add -A`) and pushes to main. Per [[commit-push-each-wave]].

---

## Track 1 — The Senior Mindset & Problem-Solving Framework (6)
1. How a senior approaches an ambiguous design problem (clarify → constrain → sketch → stress → iterate)
2. From vague ask to requirements: functional vs non-functional, SLAs/SLOs, scoping
3. Back-of-the-envelope estimation: QPS, storage, bandwidth — fast and defensible
4. Articulating tradeoffs: CAP, PACELC, latency vs throughput, cost vs reliability — and defending a call
5. Diagrams that communicate: C4 model, sequence diagrams, knowing when to zoom
6. Evolutionary architecture: designing for change, premature-abstraction cost, "build for 10x not 1000x"

## Track 2 — Foundational Building Blocks at Tradeoff Depth (12)
7. Storage engines: B-trees vs LSM-trees, when each wins
8. SQL vs NoSQL vs NewSQL: choosing a datastore by access pattern
9. Replication: leader-follower, multi-leader, leaderless — and failure modes
10. Partitioning/sharding: hash vs range, hot keys, resharding without downtime
11. Consistency models: linearizable → causal → eventual, and what each costs
12. Consensus & coordination: Raft/Paxos, leader election, when you actually need etcd/ZooKeeper
13. Caching: cache-aside vs write-through, invalidation, stampedes, multi-tier
14. Load balancing: L4 vs L7, algorithms, health checks, consistent hashing
15. Queues & event streaming: queue vs log, delivery semantics, the outbox
16. API design: REST vs gRPC vs GraphQL, pagination, versioning
17. Idempotency & exactly-once: safe retries the Stripe way
18. Rate limiting & backpressure: token bucket, sliding window, distributed counters, load shedding

## Track 3 — Cross-Cutting Concerns (8)
19. Reliability: SLI/SLO/error budgets, redundancy, graceful degradation
20. Observability: metrics, logs, traces — designing for debuggability
21. Capacity planning & autoscaling: predicting, provisioning, the cost of headroom
22. Security at the architecture level: authn/authz, secrets, zero-trust, defense in depth
23. Multi-region & geo-distribution: active-active vs active-passive, failover, data residency
24. Cost as a design constraint: the FinOps mindset, when cheaper beats faster
25. Schema & API evolution: migrations at scale, expand-contract, compatibility
26. Testing distributed systems: chaos engineering, load testing, fault injection

## Track 4 — Professional "Design X" Case Studies (8)
27. Design a URL shortener (staff-depth warm-up)
28. Design a news feed / timeline (fan-out write vs read, the celebrity problem) — Twitter/Instagram
29. Design a chat/messaging system (delivery, presence, ordering) — WhatsApp/Discord
30. Design ride-hailing / geospatial matching — Uber/Lyft
31. Design a payment system (ledgers, idempotency, reconciliation) — Stripe
32. Design video streaming (CDN, transcoding, adaptive bitrate) — YouTube/Netflix
33. Design search / autocomplete (inverted index, ranking, typeahead)
34. Design a collaborative editor (CRDT/OT, presence) — Figma/Google Docs

## Track 5 — Failure, Post-Mortems & Debugging at Scale (4)
35. Anatomy of an outage: patterns from real public post-mortems (S3, Cloudflare, GitHub, Slack)
36. Cascading failures, retry storms, thundering herds — circuit breakers & bulkheads
37. Debugging production at scale: USE/RED, four golden signals, correlating signals
38. The hard problems: distributed transactions, dual writes, outbox, sagas

## Track 6 — End-to-End Capstones (2)
39. Capstone: designing a complete system end-to-end (multi-tenant SaaS analytics platform)
40. The senior's design-review checklist & anti-patterns: a synthesis

---

## Status
- [x] Wave 1 — Track 1 (posts 1-6) — SHIPPED commit 4f828cd (66k words, 54 figs, all gates green)
- [x] Wave 2 — Track 2a (posts 7-12) — SHIPPED commit 316169e (68k words, 54 figs, all gates green)
- [x] Wave 3 — Track 2b (posts 13-18) — SHIPPED commit 37b58c9 (70k words, 54 figs, all gates green)
  → Waves 1-3 COMPLETE (18 posts). Remaining: Wave 4 (19-26 cross-cutting), Wave 5 (27-34 case studies), Wave 6 (35-40 failure+capstone).
- [x] Wave 4 — Track 3 (posts 19-26) — SHIPPED commit 9f6b4fc (93k words, 72 figs, all gates green)
- [x] Wave 5 — Track 4 (posts 27-34) — SHIPPED commit be5fb12 (90k words, 72 figs, all gates green)
- [x] Wave 6 — Track 5+6 (posts 35-40) — SHIPPED commit d3eeec7 (70k words, 54 figs, all gates green)

## SERIES COMPLETE — all 40 posts shipped+pushed to main 2026-06-15
~464k words, 360 figures. Wave commits: 4f828cd / 316169e / 37b58c9 / 9f6b4fc / be5fb12 / d3eeec7.
Final 40-post cross-link audit: 0 broken / 71 distinct links.
