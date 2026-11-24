---
title: "GPU Utilization Is a Poor Autoscaling Signal"
publishDate: "2026-07-31"
date: "2026-07-31"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Together AI explains that GPU utilization can look moderate while an inference engine's queue is already backing up, so it misleads autoscaling."
---

> Source: [@togethercompute on X](https://x.com/togethercompute/status/2083318555681177640) · posted 2026-07-31

## Summary

Together AI says the first lesson from building autoscaling for dedicated inference is that CPU-style metrics are not enough. A GPU can show 60% busy while the engine's request queue is already growing. They explain that utilization measures arithmetic intensity, not pressure.

## Background

Autoscaling adds or removes servers based on load. GPU utilization only shows how much of the time the GPU is doing work, not whether requests are waiting. Signals like queue depth, waiting time or KV cache usage better reflect whether users are about to see slow responses.

## What I took away

- Scale inference on queue length or latency signals, not only GPU utilization.
- A partly busy GPU can still be the bottleneck.
- Define the user-facing metric (like time to first token) first, then pick scaling signals.

## Reference

- Original post: [https://x.com/togethercompute/status/2083318555681177640](https://x.com/togethercompute/status/2083318555681177640)
- Author: [@togethercompute](https://x.com/togethercompute) on X, posted 2026-07-31
