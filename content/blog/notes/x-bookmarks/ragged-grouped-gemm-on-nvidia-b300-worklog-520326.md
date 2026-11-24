---
title: "Ragged Grouped GEMM on NVIDIA B300 Worklog"
publishDate: "2026-08-04"
date: "2026-08-04"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "moe", "hardware", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A recommendation of a worklog about building and profiling a ragged-M grouped GEMM on an NVIDIA B300 for MoE workloads."
---

> Source: [@MainzOnX on X](https://x.com/MainzOnX/status/2084726340096520326) · posted 2026-08-04

## Summary

The author recommends the quoted worklog, saying even they learned a lot. In it, a developer describes spending weeks building and profiling a ragged-M grouped GEMM on an NVIDIA B300 to understand the Blackwell SM, especially how irregular MoE workloads interact with tiling, scheduling, occupancy and data movement.

## Background

A grouped GEMM runs many matrix multiplies of different sizes in one launch. In MoE, each expert receives a different number of tokens, so the M dimension is ragged (uneven). The SM (streaming multiprocessor) is the GPU's core compute unit, and occupancy is how many threads it keeps active.

## What I took away

- Uneven expert loads make MoE kernels harder to optimize than dense matmul.
- Building a kernel yourself is a strong way to learn a new GPU architecture.
- Track tiling, scheduling and data movement together; they interact.

## Reference

- Original post: [https://x.com/MainzOnX/status/2084726340096520326](https://x.com/MainzOnX/status/2084726340096520326)
- Author: [@MainzOnX](https://x.com/MainzOnX) on X, posted 2026-08-04
