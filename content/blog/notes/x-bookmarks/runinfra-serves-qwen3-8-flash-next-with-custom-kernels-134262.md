---
title: "RunInfra Serves Qwen3.8 Flash Next With Custom Kernels"
publishDate: "2026-08-30"
date: "2026-08-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "gpu-kernels", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "RunInfra says it serves Qwen3.8 Flash Next fast and cheaply by writing better CUDA kernels on standard NVIDIA GPUs."
---

> Source: [@Akashi203 on X](https://x.com/Akashi203/status/2094169028923134262) · posted 2026-08-30

## Summary

The author says RunInfra does not use custom chips and instead writes better CUDA kernels on the same NVIDIA GPUs others rent. The quoted post says **Qwen3.8 Flash Next** is live on RunInfra at 249.8 tok/s output, 1,196 ms to the first reasoning token, $0.12 per 1M input and $0.40 per 1M output tokens. It adds cached input costs $0.01 per 1M with about a 95% cache hit rate, plus 1M token context and an OpenAI-compatible API.

## Background

CUDA kernels are the low-level GPU programs that do the actual math in inference. Better kernels can raise speed without new hardware. Cached input pricing means reused prompt prefixes are served from stored KV cache, so repeated context is cheap.

## What I took away

- Software and kernel work can rival custom hardware for serving speed.
- Prompt caching hit rate strongly affects real cost for agent workloads.
- Speed and price numbers are the provider's own claims; benchmark with your workload.

## Reference

- Original post: [https://x.com/Akashi203/status/2094169028923134262](https://x.com/Akashi203/status/2094169028923134262)
- Author: [@Akashi203](https://x.com/Akashi203) on X, posted 2026-08-30
