---
title: "KVzap KV Cache Pruning"
publishDate: "2026-01-14"
date: "2026-01-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2011528243413651539-1.jpg"
excerpt: "NVIDIA introduces KVzap, an input-dependent KV cache pruning method claiming 2-4x near-lossless compression."
---

> Source: [@NVIDIAAIDev on X](https://x.com/NVIDIAAIDev/status/2011528243413651539) · posted 2026-01-14

![KVzap KV Cache Pruning (1)](/imgs/blogs/x-bookmarks/2011528243413651539-1.jpg)

![KVzap KV Cache Pruning (2)](/imgs/blogs/x-bookmarks/2011528243413651539-2.png)

## Summary

NVIDIA presents KVzap, which learns to discard KV cache entries that will not be needed later, claiming 2-4x compression with negligible loss. The quoted author says it is open source and the first pruning method since KVpress worth considering for vLLM, SGLang or TRT-LLM.

## Background

The KV cache stores attention keys and values for past tokens and grows with context length. Pruning unneeded entries reduces memory and speeds up decoding.

## What I took away

- Not all cached tokens matter equally.
- Pruning could be integrated into serving engines.

## Reference

- Original post: [https://x.com/NVIDIAAIDev/status/2011528243413651539](https://x.com/NVIDIAAIDev/status/2011528243413651539)
- Author: [@NVIDIAAIDev](https://x.com/NVIDIAAIDev) on X, posted 2026-01-14
