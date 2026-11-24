---
title: "DeepGEMM Ascend Open Sourced"
publishDate: "2026-09-30"
date: "2026-09-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "hardware", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "DeepSeek open-sourced DeepGEMM Ascend, claiming 99.8% of hardware limit on GEMM."
---

> Source: [@zephyr_z9 on X](https://x.com/zephyr_z9/status/2105189355962958078) · posted 2026-09-30

## Summary

The author reacts with BRUH. The quoted post says DeepGEMM Ascend is now open source and claims 99.8% of the hardware limit on GEMM and 98% on MegaMoE, linking the GitHub repo.

## Background

GEMM is general matrix multiplication, the core operation in neural networks. Ascend is Huawei's AI accelerator line, so this brings DeepSeek's optimized kernels to non-NVIDIA hardware.

## What I took away

- Near-peak kernels on Ascend make it a more serious alternative to NVIDIA.
- Hardware-limit percentages are the authors' claims; check the benchmark setup.

## Reference

- Original post: [https://x.com/zephyr_z9/status/2105189355962958078](https://x.com/zephyr_z9/status/2105189355962958078)
- Author: [@zephyr_z9](https://x.com/zephyr_z9) on X, posted 2026-09-30
