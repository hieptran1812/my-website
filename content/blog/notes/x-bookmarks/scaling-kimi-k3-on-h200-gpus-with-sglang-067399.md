---
title: "Scaling Kimi K3 on H200 GPUs With SGLang"
publishDate: "2026-08-01"
date: "2026-08-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "Byron Hsu shares lessons from tuning Kimi K3 for throughput workloads on H200s with SGLang, starting with TP plus EP beating DP plus EP."
---

> Source: [@hsu_byron on X](https://x.com/hsu_byron/status/2083367378877067399) · posted 2026-08-01

## Summary

The author shares what was learned tuning **Kimi K3** for throughput-oriented workloads on H200 GPUs using SGLang. He notes most discussion focuses on Blackwell, so H200 lessons are useful. The first finding is that TP+EP beat DP+EP for K3, with a note that only about a quarter of its layers relate to the reason (the text is cut off).

## Background

TP (tensor parallelism) splits each layer's math across GPUs; DP (data parallelism) gives each GPU its own copy handling different requests; EP (expert parallelism) spreads MoE experts across GPUs. The best mix depends on the model architecture. H200 is the previous NVIDIA generation before Blackwell and is still widely deployed.

## What I took away

- The best parallelism strategy is model-specific; benchmark TP+EP and DP+EP rather than assuming.
- Older GPUs like H200 still matter for real deployments.
- Read the full thread for the remaining tuning lessons.

## Reference

- Original post: [https://x.com/hsu_byron/status/2083367378877067399](https://x.com/hsu_byron/status/2083367378877067399)
- Author: [@hsu_byron](https://x.com/hsu_byron) on X, posted 2026-08-01
