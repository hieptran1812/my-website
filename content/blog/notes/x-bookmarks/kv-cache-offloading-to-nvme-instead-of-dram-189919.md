---
title: "KV Cache Offloading to NVMe Instead of DRAM"
publishDate: "2026-11-23"
date: "2026-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "serving", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124932967827189919-1.jpg"
excerpt: "A post reports that vLLM showed an NVMe pool matching a DRAM pool within about 1% throughput for KV cache offloading on Kimi-K3."
---

> Source: [@YuvrajInfer on X](https://x.com/YuvrajInfer/status/2124932967827189919) · posted 2026-11-23

![KV Cache Offloading to NVMe Instead of DRAM](/imgs/blogs/x-bookmarks/2124932967827189919-1.jpg)

## Summary

The author says vLLM showed you can skip the DRAM tier for KV cache offloading. On Kimi-K3 with 16 AMD MI355X GPUs and MoRI UMBP, a 1.8 TB NVMe pool matched a 1.8 TB DRAM pool within about 1% throughput at 48, 70 and 96 concurrent sessions. The post also mentions a larger 4.6 TB NVMe pool for when capacity matters.

## Background

The KV cache stores attention keys and values for past tokens so the model does not recompute them. When GPU memory is full, caches can be offloaded to slower storage such as CPU RAM (DRAM) or SSDs (NVMe). If NVMe is nearly as fast as DRAM for this workload, serving long agent sessions gets cheaper.

## What I took away

- Offloaded KV cache can sit on NVMe with little throughput loss in this reported test.
- Results are from one setup, so measure on your own workload.
- Large capacity pools matter for long agentic sessions.

## Reference

- Original post: [https://x.com/YuvrajInfer/status/2124932967827189919](https://x.com/YuvrajInfer/status/2124932967827189919)
- Author: [@YuvrajInfer](https://x.com/YuvrajInfer) on X, posted 2026-11-23
