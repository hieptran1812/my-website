---
title: "HySparse2 Attention Architecture for MiMo-V3"
publishDate: "2026-09-23"
date: "2026-09-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "kv-cache", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Xiaomi's MiMo-V3 gets HySparse2, an architecture claimed to reduce prefill compute and KV cache while improving long-context retrieval."
---

> Source: [@_LuoFuli on X](https://x.com/_LuoFuli/status/2102766365190901957) · posted 2026-09-23

## Summary

The author says MiMo-V3 is getting a new architecture whose core, **HySparse2**, is released today. They claim less prefill compute, a smaller KV cache and better long-context retrieval all at once. Compared to MiMo-V2.6's Hybrid SWA design, they report 5.02x lower prefill FLOPs at 1M tokens, with more numbers cut off in the post.

## Background

Prefill is the step where the model processes the whole prompt before generating. SWA (sliding window attention) only attends to recent tokens to save compute. Sparse attention picks a subset of tokens to attend to. These designs try to make million-token contexts affordable without losing the ability to retrieve details from far back.

## What I took away

- Long-context efficiency usually trades off against retrieval quality; this claims to improve both.
- Track prefill FLOPs and KV cache size separately when comparing architectures.
- Read the full post or paper for the remaining numbers.

## Reference

- Original post: [https://x.com/_LuoFuli/status/2102766365190901957](https://x.com/_LuoFuli/status/2102766365190901957)
- Author: [@_LuoFuli](https://x.com/_LuoFuli) on X, posted 2026-09-23
