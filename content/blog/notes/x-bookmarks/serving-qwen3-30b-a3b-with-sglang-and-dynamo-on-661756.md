---
title: "Serving Qwen3-30B-A3B With SGLang and Dynamo on Two A100s"
publishDate: "2026-08-04"
date: "2026-08-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "Qwen shares an ambassador's project benchmarking Qwen3-30B-A3B with SGLang and Dynamo on two A100 GPUs."
---

> Source: [@QwenDevs on X](https://x.com/QwenDevs/status/2084571516910661756) · posted 2026-08-04

## Summary

Qwen's developer account shares a practical project by one of its ambassadors on serving **Qwen3-30B-A3B** with SGLang and Dynamo on 2x A100. In the quoted post, the author explains the experiments use aiperf at concurrency 32, with 500 measured requests, 64 warmup requests and 32 reusable 8k-token prompts.

## Background

Qwen3-30B-A3B is an MoE model with 30B total parameters and about 3B active per token. SGLang is a serving engine, Dynamo is NVIDIA's framework for distributed inference, and aiperf is a load-testing tool. Warmup requests let caches and compilation settle before measuring.

## What I took away

- Document benchmark settings such as concurrency, warmup and prompt reuse.
- Reusable prompts test prefix caching, which can change results a lot.
- Mid-size MoE models can be served on modest GPU setups.

## Reference

- Original post: [https://x.com/QwenDevs/status/2084571516910661756](https://x.com/QwenDevs/status/2084571516910661756)
- Author: [@QwenDevs](https://x.com/QwenDevs) on X, posted 2026-08-04
