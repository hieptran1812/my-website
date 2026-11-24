---
title: "Why SGLang Ships Its Own All-Reduce Kernel"
publishDate: "2026-08-16"
date: "2026-08-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "gpu-kernels", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Part of a series on SGLang tensor parallelism, explaining why it uses a custom all-reduce instead of NCCL."
---

> Source: [@thisispiyushK on X](https://x.com/thisispiyushK/status/2089006172615704820) · posted 2026-08-16

## Summary

The author asks why SGLang has its own all-reduce kernel when NCCL exists. A 60-layer model runs about 120 all-reduce operations per generated token, and during decode the payload is small, about 8KB at batch size 1 and typically tens of KB. The quoted earlier post in the series explains how SGLang splits fused weights, like the combined gate and up projection in SwiGLU MLPs, by overloading weight loading rather than adding logic to model code.

## Background

Tensor parallelism splits each layer's weights across GPUs, so after certain layers the GPUs must sum their partial results with an all-reduce. NCCL is NVIDIA's general communication library, tuned for large messages; for many tiny messages its fixed latency dominates, so a custom low-latency kernel can be faster. SwiGLU is a common MLP design where two projections are often fused into one matrix multiply.

## What I took away

- For small messages, latency matters more than bandwidth in communication.
- Count collective operations per token to estimate communication overhead.
- Fused weights need special handling when sharding a model across GPUs.

## Reference

- Original post: [https://x.com/thisispiyushK/status/2089006172615704820](https://x.com/thisispiyushK/status/2089006172615704820)
- Author: [@thisispiyushK](https://x.com/thisispiyushK) on X, posted 2026-08-16
