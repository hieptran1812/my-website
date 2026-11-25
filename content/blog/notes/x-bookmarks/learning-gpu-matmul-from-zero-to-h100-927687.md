---
title: "Learning GPU Matmul From Zero To H100"
publishDate: "2025-12-19"
date: "2025-12-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "learning-resources", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001965936488927687-1.jpg"
excerpt: "mertunsal2020 points to Aleksa Gordic's guide on high-performance matmul kernels on NVIDIA GPUs."
---

> Source: [@mertunsal2020 on X](https://x.com/mertunsal2020/status/2001965936488927687) · posted 2025-12-19

![Learning GPU Matmul From Zero To H100](/imgs/blogs/x-bookmarks/2001965936488927687-1.jpg)

## Summary

The author had planned a blog that takes a reader from zero GPU knowledge to a cuBLAS-level matrix multiply on H100, but found that Aleksa Gordic had already made great material. The link card is Inside NVIDIA GPUs: Anatomy of high performance matmul kernels.

## Background

Matmul means matrix multiplication, the main operation in neural networks. cuBLAS is NVIDIA's tuned library for it. Matching it on an H100 requires understanding memory hierarchy, tiling and tensor cores, so a guided path is very useful.

## What I took away

- Check for existing excellent resources before writing your own.
- Matmul kernel optimization is a standard path into GPU programming.
- Read Gordic's article for a step-by-step route to H100 performance.

## Reference

- Original post: [https://x.com/mertunsal2020/status/2001965936488927687](https://x.com/mertunsal2020/status/2001965936488927687)
- Author: [@mertunsal2020](https://x.com/mertunsal2020) on X, posted 2025-12-19
