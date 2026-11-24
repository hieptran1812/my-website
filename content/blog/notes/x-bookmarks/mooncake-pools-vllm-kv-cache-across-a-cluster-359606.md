---
title: "Mooncake Pools vLLM KV Cache Across a Cluster"
publishDate: "2026-08-10"
date: "2026-08-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Mooncake shares KV cache across vLLM replicas, reportedly raising prefix cache hits sharply on agentic traces."
---

> Source: [@RedHat_AI on X](https://x.com/RedHat_AI/status/2086832585972359606) · posted 2026-08-10

## Summary

Red Hat AI explains that every vLLM replica has its own KV cache, so scaling out behind a load balancer makes prefix reuse collapse. Mooncake pools caches cluster-wide. On agentic traces, they report cache hits going from 1.7% to 92.2%, TTFT down 46 times and latency down 8.6 times, as presented at vLLM Office Hours #55.

## Background

The KV cache stores attention data for tokens already processed, so repeated prefixes do not need recomputing. With many replicas, a request may land on a replica that has not seen its prefix. A shared cache pool lets any replica reuse prefixes. TTFT is time to first token, the delay before output starts.

## What I took away

- Load balancing can destroy cache locality; consider cache-aware routing or shared caches.
- Agentic workloads with long repeated contexts benefit most from prefix reuse.
- Measure cache hit rate as a first-class serving metric.

## Reference

- Original post: [https://x.com/RedHat_AI/status/2086832585972359606](https://x.com/RedHat_AI/status/2086832585972359606)
- Author: [@RedHat_AI](https://x.com/RedHat_AI) on X, posted 2026-08-10
