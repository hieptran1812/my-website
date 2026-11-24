---
title: "TokenSpeed Optimizes Kimi K3 on GB300"
publishDate: "2026-09-08"
date: "2026-09-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "TokenSpeed, a young open source inference engine, shares TP8 optimization of Kimi K3 on GB300, running in production for weeks."
---

> Source: [@zhyncs42 on X](https://x.com/zhyncs42/status/2097339535608905734) · posted 2026-09-08

## Summary

The author calls it a milestone for **TokenSpeed**, an open source LLM inference engine less than 6 months old, thanking NVIDIA and Jensen Huang for engineering and GPU access support. The quoted post covers part I of Kimi K3 optimization: TP8 on GB300. It has been stable in production for 3+ weeks across about 1,000 B300 and 2,000 H200 GPUs.

## Background

TP8 means tensor parallelism across 8 GPUs, splitting each layer's math among them. GB300 NVL72 is a rack of 72 tightly connected Blackwell Ultra GPUs. Production stability over weeks is a stronger signal than a one-off benchmark.

## What I took away

- Vendor support and hardware access are key for new inference engines, as the same author argues elsewhere.
- Read the blog for practical TP tuning on Blackwell systems.
- Value long production runs over short benchmark numbers.

## Reference

- Original post: [https://x.com/zhyncs42/status/2097339535608905734](https://x.com/zhyncs42/status/2097339535608905734)
- Author: [@zhyncs42](https://x.com/zhyncs42) on X, posted 2026-09-08
