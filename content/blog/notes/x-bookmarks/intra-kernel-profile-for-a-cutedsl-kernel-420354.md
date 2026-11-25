---
title: "Intra-Kernel Profile for a CuTeDSL Kernel"
publishDate: "2026-05-27"
date: "2026-05-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "quantization", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2059640314634420354-1.jpg"
excerpt: "Maharshii gets an intra-kernel profile of a CuTeDSL kernel and quotes a comparison of NVFP4 GEMM against cuBLAS."
---

> Source: [@maharshii on X](https://x.com/maharshii/status/2059640314634420354) · posted 2026-05-27

![Intra-Kernel Profile for a CuTeDSL Kernel (1)](/imgs/blogs/x-bookmarks/2059640314634420354-1.jpg)

![Intra-Kernel Profile for a CuTeDSL Kernel (2)](/imgs/blogs/x-bookmarks/2059640314634420354-2.png)

## Summary

The author got an intra-kernel profile for a CuTeDSL kernel and jokes he now has to understand it. The quoted post compares their CuTeDSL NVFP4 GEMM kernel with cuBLAS via torch. It notes the torch scaled mm API lacks a global scale dequantization argument, which lowers FLOPS, and the author is unsure why FLOPS are low, maybe power throttling.

## Background

CuTeDSL is a Python DSL for writing NVIDIA GPU kernels. NVFP4 is a 4-bit floating-point format. GEMM is matrix multiplication. Intra-kernel profiling shows time spent inside a kernel.

## What I took away

- Profile inside the kernel to explain low FLOPS.
- API gaps can force extra work and cost speed.
- Consider power throttling when benchmarking.

## Reference

- Original post: [https://x.com/maharshii/status/2059640314634420354](https://x.com/maharshii/status/2059640314634420354)
- Author: [@maharshii](https://x.com/maharshii) on X, posted 2026-05-27
