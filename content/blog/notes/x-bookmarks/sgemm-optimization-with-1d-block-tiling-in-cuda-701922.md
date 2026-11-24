---
title: "SGEMM Optimization With 1D Block Tiling in CUDA"
publishDate: "2026-08-06"
date: "2026-08-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "Day 28 of a learning log on optimizing single-precision matrix multiply in CUDA using 1D block tiling."
---

> Source: [@maxxfuu on X](https://x.com/maxxfuu/status/2085245154202701922) · posted 2026-08-06

## Summary

On day 28 of a 90-day inference engineering log, the author learned to optimize an SGEMM kernel with 1D block tiling in CUDA. Building on shared memory block tiling, 1D block tiling raises arithmetic intensity by doing more computation per byte loaded.

## Background

SGEMM is single-precision general matrix multiplication. Arithmetic intensity is the ratio of math operations to bytes moved from memory. In 1D block tiling each thread computes several output elements, reusing loaded data from registers, so the kernel becomes less memory-bound.

## What I took away

- Raising arithmetic intensity is the main lever for faster matmul kernels.
- Work through matmul optimizations step by step to build GPU intuition.
- Each thread computing more outputs reduces memory traffic per result.

## Reference

- Original post: [https://x.com/maxxfuu/status/2085245154202701922](https://x.com/maxxfuu/status/2085245154202701922)
- Author: [@maxxfuu](https://x.com/maxxfuu) on X, posted 2026-08-06
