---
title: "SGEMM Optimization With Shared Memory Tiling"
publishDate: "2026-08-05"
date: "2026-08-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Day 27 of a learning log compares global memory coalescing with shared memory tiling for a CUDA SGEMM kernel."
---

> Source: [@maxxfuu on X](https://x.com/maxxfuu/status/2084896160691765387) · posted 2026-08-05

## Summary

On day 27 of a 90-day inference engineering log, the author optimizes an SGEMM CUDA kernel with shared memory (SMEM) tiling. They explain the difference from the previous optimization, global memory coalescing, which improves how threads in a warp access global memory.

## Background

Coalescing means threads in a warp read neighboring addresses so the GPU can combine them into fewer memory transactions. Shared memory tiling loads a block of the input matrices into fast on-chip memory once and reuses it many times, reducing slow global memory reads.

## What I took away

- Coalescing fixes access patterns; tiling fixes data reuse.
- Shared memory is small but much faster than global memory.
- Step-by-step kernel tuning is a good way to learn GPU performance.

## Reference

- Original post: [https://x.com/maxxfuu/status/2084896160691765387](https://x.com/maxxfuu/status/2084896160691765387)
- Author: [@maxxfuu](https://x.com/maxxfuu) on X, posted 2026-08-05
