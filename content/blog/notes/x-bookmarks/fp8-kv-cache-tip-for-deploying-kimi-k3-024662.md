---
title: "FP8 KV Cache Tip for Deploying Kimi K3"
publishDate: "2026-08-01"
date: "2026-08-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "quantization", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "Banghua Zhu notes SGLang defaults to bf16 KV cache for Kimi K3 and suggests testing fp8 KV cache for a large speedup if accuracy holds."
---

> Source: [@BanghuaZ on X](https://x.com/BanghuaZ/status/2083402255446024662) · posted 2026-08-01

## Summary

The author says SGLang defaults to a **bf16 KV cache** for Kimi K3 for better precision. If you test an **fp8 KV cache** and see no accuracy drop in your setup, it should greatly speed up inference. The quoted post is Byron Hsu's thread on scaling Kimi K3 on H200s with SGLang.

## Background

The KV cache stores attention keys and values for past tokens so they are not recomputed. Storing it in fp8 (8-bit) instead of bf16 (16-bit) halves memory, allowing more or longer requests and faster memory reads. The trade-off is possible accuracy loss, which depends on the model and task.

## What I took away

- Defaults often favor safety over speed; check whether they fit your workload.
- Always measure accuracy on your own tasks before switching to fp8 KV cache.
- KV cache precision is a cheap lever for throughput.

## Reference

- Original post: [https://x.com/BanghuaZ/status/2083402255446024662](https://x.com/BanghuaZ/status/2083402255446024662)
- Author: [@BanghuaZ](https://x.com/BanghuaZ) on X, posted 2026-08-01
