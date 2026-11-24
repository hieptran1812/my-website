---
title: "SGLang Tensor Parallelism and CUDA Graph Activations"
publishDate: "2026-08-23"
date: "2026-08-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "gpu-kernels", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Part 18 of an SGLang series explains how tensor-parallel processes handle activations and memory in CUDA graphs."
---

> Source: [@thisispiyushK on X](https://x.com/thisispiyushK/status/2091596432256229586) · posted 2026-08-23

## Summary

Part 18 of the author's SGLang series covers tensor parallelism activations in CUDA graphs. It explains that in TP each GPU runs a separate process with its own private page table, and that during inference the intermediate allocations for all-reduce operations must be shared among them, where the post is cut off. The quoted part 17 covers CPU pinning in multi-GPU setups: SGLang starts one Python process per GPU, and each GPU connects to one CPU socket.

## Background

Tensor parallelism splits each layer's matrices across GPUs, which then combine results with an all-reduce. CUDA graphs record a fixed sequence of GPU operations to replay with low overhead, which requires stable memory addresses. Pinning a process to the CPU near its GPU avoids slow cross-socket memory traffic.

## What I took away

- Pin GPU processes to the CPU socket they are attached to.
- CUDA graphs need fixed memory addresses, which complicates shared buffers across processes.
- Reading engine internals series is a good path into serving engineering.

## Reference

- Original post: [https://x.com/thisispiyushK/status/2091596432256229586](https://x.com/thisispiyushK/status/2091596432256229586)
- Author: [@thisispiyushK](https://x.com/thisispiyushK) on X, posted 2026-08-23
