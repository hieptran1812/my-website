---
title: "Making Hybrid Sliding Window Attention Work in Production"
publishDate: "2026-07-18"
date: "2026-07-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "llm-architecture", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A paper on how Xiaomi fixed its KV cache system so hybrid sliding window attention cuts long-context cost for MiMo-V2.5."
---

> Source: [@askalphaxiv on X](https://x.com/askalphaxiv/status/2078484156187599103) · posted 2026-07-18

## Summary

The post says hybrid sliding window attention (SWA) can make long-context inference about 7x cheaper in theory. It highlights a paper on how Xiaomi made this work in production for MiMo-V2.5. The main trick is changing the KV cache system so SWA layers only store the recent window, while still allowing prefix reuse.

## Background

Sliding window attention lets each token attend only to a fixed number of recent tokens, so those layers need far less memory. A hybrid model mixes SWA layers with full-attention layers. The KV cache stores past keys and values; most serving systems assume every layer keeps the full context, so saving memory requires redesigning the cache, especially for prefix caching where shared prompt prefixes are reused across requests.

## What I took away

- Theoretical savings from an architecture only appear if the serving stack is built to exploit them.
- Per-layer cache policies (full versus windowed) are a key design point for hybrid models.
- Prefix reuse and windowed caches can conflict; check how a system reconciles them.

## Reference

- Original post: [https://x.com/askalphaxiv/status/2078484156187599103](https://x.com/askalphaxiv/status/2078484156187599103)
- Author: [@askalphaxiv](https://x.com/askalphaxiv) on X, posted 2026-07-18
