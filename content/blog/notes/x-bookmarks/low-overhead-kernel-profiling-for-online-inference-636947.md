---
title: "Low-Overhead Kernel Profiling for Online Inference"
publishDate: "2026-11-04"
date: "2026-11-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A preview of a method to track per-kernel timing in every step of live inference with under 1% overhead."
---

> Source: [@SuJinYan123 on X](https://x.com/SuJinYan123/status/2117929517465636947) · posted 2026-11-04

## Summary

@SuJinYan123 previews a way to observe kernel performance in online inference with reasonable overhead. It tracks the time of every kernel in every step while keeping end-to-end performance impact below 1%. The author believes no one has done this before.

## Background

A kernel is a function that runs on the GPU, and one inference step launches many of them. Profilers usually add heavy overhead, so they are used offline. Measuring kernels in production lets engineers see real bottlenecks under actual traffic.

## What I took away

- Production profiling can reveal bottlenecks that offline benchmarks miss.
- Overhead is the key constraint for always-on monitoring.
- Per-step kernel timing helps find regressions after deployments.

## Reference

- Original post: [https://x.com/SuJinYan123/status/2117929517465636947](https://x.com/SuJinYan123/status/2117929517465636947)
- Author: [@SuJinYan123](https://x.com/SuJinYan123) on X, posted 2026-11-04
