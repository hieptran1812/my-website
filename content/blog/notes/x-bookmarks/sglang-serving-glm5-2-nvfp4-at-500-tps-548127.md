---
title: "SGLang Serving GLM5.2 NVFP4 at 500 TPS"
publishDate: "2026-07-14"
date: "2026-07-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "serving", "llm-inference", "quantization", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "LMSYS explains how SGLang reached over 500 tokens per second per user at batch size 1 on 8xB300 for GLM5.2."
---

> Source: [@lmsysorg on X](https://x.com/lmsysorg/status/2077076059657548127) · posted 2026-07-14

## Summary

LMSYS publishes a deep dive on serving GLM5.2 in NVFP4 for agentic workloads with SGLang. They report 500+ tokens per second per user at batch size 1 on 8xB300. They say this is 18 to 34% higher single-user interactivity within two weeks of day-0 support, plus 6 to 11% better peak results; the text cuts off.

## Background

Batch size 1 measures speed for a single user, which matters for interactive agents. NVFP4 is NVIDIA's 4-bit floating point format, which cuts memory traffic. B300 is a recent NVIDIA GPU. Tokens per second per user reflects how fast responses stream.

## What I took away

- Single-user latency and peak throughput are different targets that need different optimizations.
- Serving engines keep improving quickly after a model's launch.
- Read the deep dive for specific optimizations to reuse.

## Reference

- Original post: [https://x.com/lmsysorg/status/2077076059657548127](https://x.com/lmsysorg/status/2077076059657548127)
- Author: [@lmsysorg](https://x.com/lmsysorg) on X, posted 2026-07-14
