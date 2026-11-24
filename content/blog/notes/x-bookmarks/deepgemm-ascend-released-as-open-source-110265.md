---
title: "DeepGEMM Ascend Released as Open Source"
publishDate: "2026-09-30"
date: "2026-09-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "hardware", "moe"]
author: "Hiep Tran"
featured: false
excerpt: "DeepGEMM Ascend, a matrix multiplication library for Huawei Ascend chips, is open source and is reported to reach near hardware limits."
---

> Source: [@zheanxu on X](https://x.com/zheanxu/status/2105119944556110265) · posted 2026-09-30

## Summary

The author announces that **DeepGEMM Ascend** from deepseek-ai is now open source. They report reaching 99.8% of the hardware limit on GEMM and 98% on MegaMoE.

## Background

GEMM is general matrix multiplication, the core operation in neural networks. Ascend is Huawei's line of AI accelerators. Reaching close to the hardware limit means the kernel uses almost all of the chip's peak compute, which directly lowers training and inference cost. MoE (mixture of experts) layers route tokens to a few expert sub-networks and are harder to run efficiently.

## What I took away

- Efficient kernels for non-NVIDIA hardware are becoming open and competitive.
- Measure kernels as a percentage of hardware peak, not only raw speed.
- MoE kernels are a separate optimization target from plain GEMM.

## Reference

- Original post: [https://x.com/zheanxu/status/2105119944556110265](https://x.com/zheanxu/status/2105119944556110265)
- Author: [@zheanxu](https://x.com/zheanxu) on X, posted 2026-09-30
