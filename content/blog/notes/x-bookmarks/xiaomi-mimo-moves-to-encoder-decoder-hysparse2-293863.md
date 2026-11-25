---
title: "Xiaomi MiMo Moves To Encoder-Decoder HySparse2"
publishDate: "2026-11-07"
date: "2026-11-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "kv-cache", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2119142144749293863-1.jpg"
excerpt: "A report that Xiaomi MiMo is adopting a causal encoder-decoder with big prefill and KV cache savings."
---

> Source: [@SemiAnalysis_ on X](https://x.com/SemiAnalysis_/status/2119142144749293863) · posted 2026-11-07

![Xiaomi MiMo Moves To Encoder-Decoder HySparse2](/imgs/blogs/x-bookmarks/2119142144749293863-1.jpg)

## Summary

The post says after DeepSeek V4.1 Flash, Xiaomi MiMo is moving to a causal encoder-decoder for next-gen models. In HySparse2, prompts run through 25 of 49 layers and decoder KV is projected from encoder states. At 1M context versus MiMo-V2, it claims 5x less prefill compute and 4.5x smaller KV cache.

## Background

Running prompts through fewer layers reduces prefill cost, and deriving decoder KV from encoder states cuts cache size. Numbers are as reported in the post.

## What I took away

- Architecture can cut long-context cost.
- Prefill and KV cache dominate at 1M tokens.
- Figures are reported claims.

## Reference

- Original post: [https://x.com/SemiAnalysis_/status/2119142144749293863](https://x.com/SemiAnalysis_/status/2119142144749293863)
- Author: [@SemiAnalysis_](https://x.com/SemiAnalysis_) on X, posted 2026-11-07
