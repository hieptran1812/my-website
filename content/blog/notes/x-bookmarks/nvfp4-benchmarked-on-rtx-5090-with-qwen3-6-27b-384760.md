---
title: "NVFP4 Benchmarked On RTX 5090 With Qwen3.6-27B"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062839600662384760-1.jpg"
excerpt: "A user benchmarks NVFP4 against Q4_K_M and Q6_K for Qwen3.6-27B on a 5090 using llama.cpp."
---

> Source: [@witcheer on X](https://x.com/witcheer/status/2062839600662384760) · posted 2026-06-05

![NVFP4 Benchmarked On RTX 5090 With Qwen3.6-27B](/imgs/blogs/x-bookmarks/2062839600662384760-1.jpg)

## Summary

The author questions the claim that NVFP4 makes Blackwell cards faster. They ran Qwen3.6-27B on a 5090 in three formats (NVFP4, Q4_K_M, Q6_K) with the same llama.cpp build and harness. They report that prefill, the prompt processing step, is clearly faster with NVFP4, about 32 to 42 percent over the comparison.

## Background

NVFP4 is a 4-bit floating point format supported natively by Blackwell GPUs. Prefill processes the whole prompt at once and is compute heavy, so hardware support for the format helps there. Decoding is different and is often limited by memory bandwidth.

## What I took away

- Benchmark prefill and decode separately.
- Compare formats at the same bit budget.
- The post is cut off, so read the full thread for decode results.

## Reference

- Original post: [https://x.com/witcheer/status/2062839600662384760](https://x.com/witcheer/status/2062839600662384760)
- Author: [@witcheer](https://x.com/witcheer) on X, posted 2026-06-05
