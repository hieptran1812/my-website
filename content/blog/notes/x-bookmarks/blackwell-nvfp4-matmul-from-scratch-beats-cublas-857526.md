---
title: "Blackwell NVFP4 Matmul From Scratch Beats cuBLAS"
publishDate: "2026-08-15"
date: "2026-08-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "quantization", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "A blog sequel implementing an NVFP4 matmul for Blackwell from scratch, reported faster than cuBLAS."
---

> Source: [@hsu_byron on X](https://x.com/hsu_byron/status/2088775194160857526) · posted 2026-08-15

## Summary

The author cheers the return of a well-known kernel blogger. The quoted post announces a sequel to an H100 matmul blog, implementing a Blackwell matmul for NVFP4 from scratch and outperforming cuBLAS by 4.7% at N=8192. It mentions a tile ordering scheme better suited to NVIDIA hardware than Hilbert curves.

## Background

A matmul (matrix multiplication) is the core operation in neural networks. NVFP4 is NVIDIA's 4-bit floating point format on Blackwell GPUs, which saves memory and boosts throughput. cuBLAS is NVIDIA's highly tuned library, so beating it is a strong result. Tile ordering, like Hilbert curves, decides the order blocks are computed to improve cache reuse.

## What I took away

- Writing a matmul from scratch is a deep way to learn GPU architecture.
- The order in which tiles are processed affects cache hits and performance.
- A small win over cuBLAS at one size may not hold at all sizes.

## Reference

- Original post: [https://x.com/hsu_byron/status/2088775194160857526](https://x.com/hsu_byron/status/2088775194160857526)
- Author: [@hsu_byron](https://x.com/hsu_byron) on X, posted 2026-08-15
