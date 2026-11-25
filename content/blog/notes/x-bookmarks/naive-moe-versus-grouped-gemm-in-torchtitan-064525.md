---
title: "Naive MoE Versus Grouped GEMM In torchtitan"
publishDate: "2025-12-06"
date: "2025-12-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "gpu-kernels", "learning-resources"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1997220320777064525-1.jpg"
excerpt: "A pointer to torchtitan code that compares a naive for-loop MoE MLP with a grouped GEMM version."
---

> Source: [@zcbenz on X](https://x.com/zcbenz/status/1997220320777064525) · posted 2025-12-06

![Naive MoE Versus Grouped GEMM In torchtitan (1)](/imgs/blogs/x-bookmarks/1997220320777064525-1.jpg)

![Naive MoE Versus Grouped GEMM In torchtitan (2)](/imgs/blogs/x-bookmarks/1997220320777064525-2.jpg)

## Summary

@zcbenz points to code in torchtitan that implements a mixture-of-experts MLP with a vanilla for-loop and with grouped GEMM. They say it is a nice way to learn why high-performance MoE needs grouped GEMM and thank the developers for leaving the naive version.

## Background

In an MoE layer, each expert processes a different set of tokens. A for-loop runs experts one at a time, which wastes the GPU. Grouped GEMM batches the many small matrix multiplications into one efficient kernel call.

## What I took away

- Reading a naive reference next to the optimized version teaches why the optimization exists.
- MoE speed depends on batching many small matmuls.

## Reference

- Original post: [https://x.com/zcbenz/status/1997220320777064525](https://x.com/zcbenz/status/1997220320777064525)
- Author: [@zcbenz](https://x.com/zcbenz) on X, posted 2025-12-06
