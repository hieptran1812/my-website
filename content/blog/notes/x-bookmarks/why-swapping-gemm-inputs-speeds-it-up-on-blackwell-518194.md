---
title: "Why Swapping GEMM Inputs Speeds It Up on Blackwell"
publishDate: "2026-08-14"
date: "2026-08-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "An explanation of why computing the transposed product of a GEMM can make it much faster on Blackwell GPUs."
---

> Source: [@jessiedong_ on X](https://x.com/jessiedong_/status/2088105178159518194) · posted 2026-08-14

## Summary

The author asks why swapping the inputs to a GEMM can make it 50% faster. They say all state-of-the-art GEMMs on Blackwell use this technique, and that DeepGEMM, the FP8 and FP4 GEMM library SGLang uses to serve DeepSeek, computes C transpose = B transpose times A transpose instead of C = A times B.

## Background

A GEMM is a general matrix multiply. Because the transpose of a product equals the product of transposes in reverse order, the same result can be computed with the inputs swapped. On GPU tensor cores, one operand dimension often has a minimum tile size, so when one matrix is small (as with few tokens in decode), swapping lets the small dimension fit the flexible side and wastes less hardware.

## What I took away

- Mathematically equivalent formulations can have very different GPU performance.
- Small batch dimensions in decode often motivate swapping GEMM operands.
- Look at DeepGEMM's code to see the technique in practice.

## Reference

- Original post: [https://x.com/jessiedong_/status/2088105178159518194](https://x.com/jessiedong_/status/2088105178159518194)
- Author: [@jessiedong_](https://x.com/jessiedong_) on X, posted 2026-08-14
