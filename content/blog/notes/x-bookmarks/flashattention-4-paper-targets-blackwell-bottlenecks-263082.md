---
title: "FlashAttention-4 Paper Targets Blackwell Bottlenecks"
publishDate: "2026-03-05"
date: "2026-03-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "hardware", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2029569881151263082-1.jpg"
excerpt: "Tri Dao announces the FA4 paper, saying attention on Blackwell GPUs now runs about as fast as matmul."
---

> Source: [@tri_dao on X](https://x.com/tri_dao/status/2029569881151263082) · posted 2026-03-05

![FlashAttention-4 Paper Targets Blackwell Bottlenecks](/imgs/blogs/x-bookmarks/2029569881151263082-1.jpg)

## Summary

The author says the FA4 paper is out after a year. On Blackwell GPUs attention now goes about as fast as matmul, even though the bottlenecks differ: forward is limited by the exponential and backward by shared memory. The quoted post says FA4 reaches about 1600 TFLOPs by changing the algorithm and pipeline.

## Background

Tensor cores do matrix multiplication; softmax needs exponentials, which run on slower units. As tensor cores get faster, the other parts become the limit, called asymmetric hardware scaling. FA4 reshapes the work to hide those costs.

## What I took away

- Faster matmul shifts bottlenecks to exp and shared memory.
- Kernel design must follow hardware changes.
- Numbers are the authors' claims for Blackwell.

## Reference

- Original post: [https://x.com/tri_dao/status/2029569881151263082](https://x.com/tri_dao/status/2029569881151263082)
- Author: [@tri_dao](https://x.com/tri_dao) on X, posted 2026-03-05
