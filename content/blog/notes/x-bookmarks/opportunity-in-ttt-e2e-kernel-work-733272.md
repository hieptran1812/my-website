---
title: "Opportunity In TTT-E2E Kernel Work"
publishDate: "2026-01-12"
date: "2026-01-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2010807997220733272-1.jpg"
excerpt: "A call for CUDA kernel experts: TTT-E2E meta-learning is 3.4x slower than pretraining at 8K context."
---

> Source: [@archiexzzz on X](https://x.com/archiexzzz/status/2010807997220733272) · posted 2026-01-12

![Opportunity In TTT-E2E Kernel Work](/imgs/blogs/x-bookmarks/2010807997220733272-1.jpg)

## Summary

The author says TTT-E2E's meta-learning needs gradients of gradients and is currently 3.4x slower than standard pretraining at 8K context because the standard FlashAttention API does not support it. The quoted NVIDIA post describes End-to-End Test-Time Training, where LLMs keep learning at test time by next-token prediction on context.

## Background

Test-time training updates weights while reading context, compressing it into the model. Second-order gradients need special attention kernels.

## What I took away

- Custom kernels are a speed bottleneck.
- Test-time training is a new memory approach.

## Reference

- Original post: [https://x.com/archiexzzz/status/2010807997220733272](https://x.com/archiexzzz/status/2010807997220733272)
- Author: [@archiexzzz](https://x.com/archiexzzz) on X, posted 2026-01-12
