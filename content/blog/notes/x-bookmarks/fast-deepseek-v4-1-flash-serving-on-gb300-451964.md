---
title: "Fast DeepSeek V4.1 Flash Serving on GB300"
publishDate: "2026-09-11"
date: "2026-09-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "gpu-kernels", "speculative-decoding", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "Praise for a team that pushed DeepSeek V4.1 Flash to 873 tok/s at batch size 1 on 4xGB300 within a day of launch."
---

> Source: [@unseenmars_ on X](https://x.com/unseenmars_/status/2098535522352451964) · posted 2026-09-11

## Summary

The author says the quoted work by @lsyincs should be printed and framed. The quoted post says that within 24 hours of launch the team reached 873 tok/s at batch size 1 on 4xGB300. They used FP8 GEMM fast paths, kernel fusion and overlap, DSpark optimization and MoE TP4, while noting DeepSeek V4.1 Flash brought major architecture changes.

## Background

FP8 is an 8-bit number format that speeds up matrix math. Kernel fusion combines several operations into one GPU kernel to cut memory traffic. TP4 means tensor parallelism over 4 GPUs, splitting each layer across them. Batch size 1 measures single-user latency.

## What I took away

- Day-0 model support is now a race among serving teams.
- Fusion, FP8 and overlap are the standard levers for low-latency decode.
- Batch size 1 numbers tell you about latency, not total throughput.

## Reference

- Original post: [https://x.com/unseenmars_/status/2098535522352451964](https://x.com/unseenmars_/status/2098535522352451964)
- Author: [@unseenmars_](https://x.com/unseenmars_) on X, posted 2026-09-11
