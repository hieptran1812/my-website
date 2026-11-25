---
title: "Compressing Qwen3.5-395B Down to 94GB"
publishDate: "2026-03-01"
date: "2026-03-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "open-models", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2028030180594266597-1.jpg"
excerpt: "A post on a study compressing Qwen3.5-395B-A17B with Unsloth quantization to fit a 128GB machine."
---

> Source: [@TeksEdge on X](https://x.com/TeksEdge/status/2028030180594266597) · posted 2026-03-01

![Compressing Qwen3.5-395B Down to 94GB (1)](/imgs/blogs/x-bookmarks/2028030180594266597-1.jpg)

![Compressing Qwen3.5-395B Down to 94GB (2)](/imgs/blogs/x-bookmarks/2028030180594266597-2.jpg)

![Compressing Qwen3.5-395B Down to 94GB (3)](/imgs/blogs/x-bookmarks/2028030180594266597-3.png)

## Summary

The author says a study compressed Qwen3.5-395B-A17B to 94GB using Unsloth's compression, so it can run on a DGX-Spark with 128GB. The quoted post evaluates GGUF variants of Qwen3.5 and says it is robust to quantization, TQ1_0 preserves accuracy well, and most degradation shows on MMLU Pro.

## Background

Quantization reduces precision to shrink memory. GGUF is a local-model file format. MMLU Pro is a hard knowledge benchmark.

## What I took away

- Heavy quantization can keep accuracy on some models.
- Check degradation on benchmarks relevant to you.
- Memory fit decides what you can run at home.

## Reference

- Original post: [https://x.com/TeksEdge/status/2028030180594266597](https://x.com/TeksEdge/status/2028030180594266597)
- Author: [@TeksEdge](https://x.com/TeksEdge) on X, posted 2026-03-01
