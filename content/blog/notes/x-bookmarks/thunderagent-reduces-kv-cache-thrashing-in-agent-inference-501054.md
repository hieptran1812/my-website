---
title: "ThunderAgent Reduces KV Cache Thrashing in Agent Inference"
publishDate: "2026-07-29"
date: "2026-07-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "serving", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "Together AI presents ThunderAgent, a scheduler-level fix for KV cache thrashing in agentic inference, accepted as an ICML 2026 Spotlight."
---

> Source: [@togethercompute on X](https://x.com/togethercompute/status/2082599087707501054) · posted 2026-07-29

## Summary

Together AI says agentic inference wastes GPUs through **KV cache thrashing**. **ThunderAgent** fixes this at the scheduler level and claims 2.5x higher single-node throughput and about 10x lower P50 latency at high concurrency. The work was accepted to ICML 2026 as a Spotlight paper.

## Background

Agents make many calls that share long context; caching that context's KV states avoids recomputation. Thrashing happens when cache entries are evicted and then needed again soon, wasting work. A scheduler that understands agent programs can keep the right entries and order requests well.

## What I took away

- Agent workloads have different cache patterns than single chat requests.
- Scheduling decisions can matter as much as kernel speed.
- P50 latency is the median; also check tail latency for user experience.

## Reference

- Original post: [https://x.com/togethercompute/status/2082599087707501054](https://x.com/togethercompute/status/2082599087707501054)
- Author: [@togethercompute](https://x.com/togethercompute) on X, posted 2026-07-29
