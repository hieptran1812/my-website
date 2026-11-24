---
title: "DeepJIT Kernel JIT Runtime for CUDA and Ascend"
publishDate: "2026-09-10"
date: "2026-09-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "hardware", "tools"]
author: "Hiep Tran"
featured: false
excerpt: "DeepSeek's DeepJIT is a lightweight header-only C++20 JIT runtime for NVIDIA CUDA GPUs and Huawei Ascend NPUs."
---

> Source: [@keeplebalive on X](https://x.com/keeplebalive/status/2097923944254767362) · posted 2026-09-10

## Summary

The post shares DeepSeek's **DeepJIT** GitHub repo. It is a lightweight, header-only C++20 JIT runtime for kernel compilation on NVIDIA CUDA GPUs and Huawei Ascend NPUs.

## Background

JIT (just-in-time) compilation builds kernels at runtime, so they can be specialized for exact shapes and hardware. Header-only means you include it without building a separate library. Supporting both CUDA and Ascend helps code run on multiple chip vendors.

## What I took away

- Runtime kernel compilation allows shape-specific optimizations.
- Cross-vendor tooling is growing as teams diversify hardware.
- Header-only libraries are easy to drop into existing projects.

## Reference

- Original post: [https://x.com/keeplebalive/status/2097923944254767362](https://x.com/keeplebalive/status/2097923944254767362)
- Author: [@keeplebalive](https://x.com/keeplebalive) on X, posted 2026-09-10
