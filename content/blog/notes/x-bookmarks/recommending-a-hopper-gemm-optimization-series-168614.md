---
title: "Recommending a Hopper GEMM Optimization Series"
publishDate: "2026-02-03"
date: "2026-02-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "learning-resources", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2018798637439168614-1.jpg"
excerpt: "The author recommends an account that documents optimizing GEMM on H100 toward cuBLAS-level performance."
---

> Source: [@suryasure05 on X](https://x.com/suryasure05/status/2018798637439168614) · posted 2026-02-03

![Recommending a Hopper GEMM Optimization Series (1)](/imgs/blogs/x-bookmarks/2018798637439168614-1.jpg)

![Recommending a Hopper GEMM Optimization Series (2)](/imgs/blogs/x-bookmarks/2018798637439168614-2.jpg)

![Recommending a Hopper GEMM Optimization Series (3)](/imgs/blogs/x-bookmarks/2018798637439168614-3.jpg)

![Recommending a Hopper GEMM Optimization Series (4)](/imgs/blogs/x-bookmarks/2018798637439168614-4.jpg)

## Summary

The author calls the quoted account a gold mine. The quoted post says they are rebuilding GEMM optimization on H100 from scratch, with 8 kernels so far up to Tensor Cores with async TMA and WGMMA, reaching 407.7 TFLOP/s or 56.9% of cuBLAS.

## Background

GEMM is general matrix multiplication, the core operation in neural nets. cuBLAS is NVIDIA's tuned library. TMA and WGMMA are Hopper features for async data movement and tensor core math.

## What I took away

- Building kernels step by step is a good way to learn GPU performance.
- Reaching library speed is hard.
- Follow the blog series to learn Hopper features.

## Reference

- Original post: [https://x.com/suryasure05/status/2018798637439168614](https://x.com/suryasure05/status/2018798637439168614)
- Author: [@suryasure05](https://x.com/suryasure05) on X, posted 2026-02-03
