---
title: "Advanced CUDA Graph Techniques for Inference Engines"
publishDate: "2026-08-17"
date: "2026-08-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "An LMSYS blog post on using CUDA Graphs in a real inference engine and the design challenges involved."
---

> Source: [@lmsysorg on X](https://x.com/lmsysorg/status/2089392627578011902) · posted 2026-08-17

## Summary

LMSYS published a blog on advanced CUDA Graph techniques in inference and the design story behind them. CUDA Graph removes kernel-launch overhead by recording GPU work once and replaying it. Bringing it into a real inference engine introduces challenges such as operator compatibility and graph management.

## Background

Each GPU kernel launch has a small CPU-side cost. During decoding, many tiny kernels run per token, so launch overhead can dominate. A CUDA Graph captures a fixed sequence of GPU operations once and replays it cheaply, but it needs fixed shapes and compatible operations, which is hard with dynamic batches.

## What I took away

- Kernel launch overhead matters most at small batch sizes during decode.
- CUDA Graphs need static shapes, so engines usually capture graphs for several batch sizes.
- Read the post for how they handle ops that do not fit graph capture.

## Reference

- Original post: [https://x.com/lmsysorg/status/2089392627578011902](https://x.com/lmsysorg/status/2089392627578011902)
- Author: [@lmsysorg](https://x.com/lmsysorg) on X, posted 2026-08-17
