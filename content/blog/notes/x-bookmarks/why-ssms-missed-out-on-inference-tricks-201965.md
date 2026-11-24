---
title: "Why SSMs Missed Out on Inference Tricks"
publishDate: "2026-10-12"
date: "2026-10-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "llm-architecture", "quantization"]
author: "Hiep Tran"
featured: false
excerpt: "Tri Dao says SSMs lacked speculative decoding and quantization because the wrong thing was cached, and ReplaySSM fixes that."
---

> Source: [@tri_dao on X](https://x.com/tri_dao/status/2109691017570201965) · posted 2026-10-12

## Summary

Tri Dao writes that SSMs had good asymptotics but kept missing inference features Transformers get for free, such as speculative decoding and quantization. He says the issue was not caching the right thing, and the fix uses the same IO-aware idea behind FlashAttention, the Mamba scan and paged KV. The quoted post reports ReplaySSM running in production in GLM-5.3-Flash and Kimi-K3 with claimed speedups up to 1.47x for decoding and 2.1x for speculative decoding.

## Background

A state space model (SSM), such as Mamba, is an alternative to attention that keeps a fixed-size running state instead of a growing KV cache. That gives good asymptotic cost for long sequences, but many inference tricks built for Transformers, like speculative decoding and low-precision caches, did not carry over easily. IO-aware means designing algorithms around the cost of moving data between memory levels rather than just counting math operations.

## What I took away

- Many big inference wins come from choosing what to store and what to recompute.
- Memory movement, not arithmetic, is often the real bottleneck.
- Pair this with the Albert Gu post on the same release.

## Reference

- Original post: [https://x.com/tri_dao/status/2109691017570201965](https://x.com/tri_dao/status/2109691017570201965)
- Author: [@tri_dao](https://x.com/tri_dao) on X, posted 2026-10-12
