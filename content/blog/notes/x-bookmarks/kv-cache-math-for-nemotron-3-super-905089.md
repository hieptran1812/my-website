---
title: "KV Cache Math For Nemotron 3 Super"
publishDate: "2026-03-11"
date: "2026-03-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "quantization"]
author: "Hiep Tran"
featured: false
excerpt: "A calculation of KV cache size per token for Nemotron 3 Super in BF16 and FP8."
---

> Source: [@bnjmn_marie on X](https://x.com/bnjmn_marie/status/2031821490916905089) · posted 2026-03-11

## Summary

With 8 attention layers, 2 KV heads and head dimension 128, the author computes the growing KV cache as 8,192 bytes per token in BF16 and 4,096 in FP8. That gives 7.63 GiB (BF16) or 3.81 GiB (FP8) at 1M tokens, and 2.00 GiB or 1.00 GiB at 262k tokens.

## Background

The KV cache stores keys and values for past tokens so the model does not recompute them. Its size is layers times KV heads times head dim times 2 times bytes per value, per token. Fewer attention layers and heads, and lower precision like FP8, shrink it a lot.

## What I took away

- KV bytes per token = 2 x layers x KV heads x head dim x bytes per value.
- FP8 halves cache size versus BF16.
- Few attention layers makes long context much cheaper.

## Reference

- Original post: [https://x.com/bnjmn_marie/status/2031821490916905089](https://x.com/bnjmn_marie/status/2031821490916905089)
- Author: [@bnjmn_marie](https://x.com/bnjmn_marie) on X, posted 2026-03-11
