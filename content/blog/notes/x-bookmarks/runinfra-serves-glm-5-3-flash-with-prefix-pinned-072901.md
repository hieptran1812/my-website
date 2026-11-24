---
title: "RunInfra Serves GLM 5.3 Flash With Prefix-Pinned Sessions"
publishDate: "2026-08-26"
date: "2026-08-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "kv-cache", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "RunInfra claims the fastest, cheapest GLM 5.3 Flash, with agent sessions pinned to replicas holding their prefix cache."
---

> Source: [@Akashi203 on X](https://x.com/Akashi203/status/2092709809225072901) · posted 2026-08-26

## Summary

The author claims RunInfra offers the fastest and cheapest **GLM 5.3 Flash**, at 254.1 tok/s and $0.40 per 1M output, and that agent sessions stay pinned to the replica holding their prefix so replays bill at $0.01 per 1M cached tokens. The quoted post adds 703 ms to first token, $0.10 per 1M input, a 92% cache hit rate and 1M token context.

## Background

Prefix caching keeps the KV cache for a prompt's beginning so later requests that share it skip recomputation. Pinning a session to one replica means its cache is always there, raising hit rate. Agent workloads resend long histories, so this cuts cost a lot.

## What I took away

- Session affinity is a simple way to raise prefix cache hit rate.
- Cached-token pricing matters most for agent workloads with long histories.
- Speed and price claims are the provider's; test them yourself.

## Reference

- Original post: [https://x.com/Akashi203/status/2092709809225072901](https://x.com/Akashi203/status/2092709809225072901)
- Author: [@Akashi203](https://x.com/Akashi203) on X, posted 2026-08-26
