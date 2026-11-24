---
title: "Why KV Cache Eviction May Not Free VRAM"
publishDate: "2026-11-01"
date: "2026-11-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "An interview question: evicting 90% of cached tokens in vLLM does not reduce VRAM usage. Why?"
---

> Source: [@_avichawla on X](https://x.com/_avichawla/status/2116861920682860761) · posted 2026-11-01

## Summary

@_avichawla poses an LLM interview question. A reasoning model served on vLLM runs out of GPU memory on long traces; you add KV cache compression and evict 90% of cached tokens, yet VRAM usage does not change and the GPU still runs out of memory. The answer is in the thread.

## Background

The KV cache stores attention keys and values for past tokens so they are not recomputed. vLLM preallocates GPU memory for the cache in fixed blocks (paged attention), so freeing tokens inside blocks may not shrink reported memory. A likely angle is that eviction must free whole blocks, or that memory is reserved up front.

## What I took away

- Know how your serving engine allocates memory before optimizing it.
- Preallocated memory pools mean usage numbers may not reflect real occupancy.
- Read the thread's answer to confirm the exact cause.

## Reference

- Original post: [https://x.com/_avichawla/status/2116861920682860761](https://x.com/_avichawla/status/2116861920682860761)
- Author: [@_avichawla](https://x.com/_avichawla) on X, posted 2026-11-01
