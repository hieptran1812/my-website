---
title: "Jenga Hybrid Memory Allocator Inspires TokenSpeed Flat KV"
publishDate: "2026-08-03"
date: "2026-08-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "The Jenga author is glad TokenSpeed replaced Radix Cache with Flat KV, drawing on hybrid memory allocator ideas."
---

> Source: [@rogerw0108 on X](https://x.com/rogerw0108/status/2084191055139033546) · posted 2026-08-03

## Summary

The author is happy to see open-source projects drawing on the hybrid memory allocator called Jenga, linking its arXiv paper and calling it their first academic contribution and ahead of its time. The quoted TokenSpeed post says it has fully removed Radix Cache and all supported models now use Flat KV.

## Background

Radix Cache, used in SGLang, stores KV cache in a tree keyed by token prefixes for reuse. Modern models mix layer types (full attention, sliding window, linear attention) with different memory needs, and a hybrid allocator like Jenga manages memory for all of them in one pool. Flat KV suggests a simpler, unified layout.

## What I took away

- Hybrid model architectures need new KV memory management designs.
- Research ideas can take time before engines adopt them.
- Read the Jenga paper to understand KV memory for mixed layer types.

## Reference

- Original post: [https://x.com/rogerw0108/status/2084191055139033546](https://x.com/rogerw0108/status/2084191055139033546)
- Author: [@rogerw0108](https://x.com/rogerw0108) on X, posted 2026-08-03
