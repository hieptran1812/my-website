---
title: "Three Ways To Do FP8 In LLM Pretraining"
publishDate: "2026-06-27"
date: "2026-06-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "pretraining", "gpu-kernels"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2070841031872987493-1.jpg"
excerpt: "A post explains FP8 pretraining approaches per-tensor, blockwise and MXFP8, which differ in how scales are attached."
---

> Source: [@ArkadiiBessonov on X](https://x.com/ArkadiiBessonov/status/2070841031872987493) · posted 2026-06-27

![Three Ways To Do FP8 In LLM Pretraining](/imgs/blogs/x-bookmarks/2070841031872987493-1.jpg)

## Summary

@ArkadiiBessonov says there are three main ways to do FP8 in LLM pretraining, and they differ mainly in how the scale is attached: per-tensor, blockwise, or MXFP8. The post notes forward plus backward is 3 matmuls (Fprop, Dgrad, Wgrad) across 3 tensor roles. The text is cut off after that.

## Background

FP8 is an 8-bit floating-point format that makes matrix multiplies faster and smaller. Because its range is tiny, values are multiplied by a scale factor. A single scale per tensor is simple, while per-block scales (blockwise or MXFP8) track local ranges better.

## What I took away

- The scale granularity is the main design choice in FP8 training.
- Each of the three matmuls and tensor roles can use different settings.

## Reference

- Original post: [https://x.com/ArkadiiBessonov/status/2070841031872987493](https://x.com/ArkadiiBessonov/status/2070841031872987493)
- Author: [@ArkadiiBessonov](https://x.com/ArkadiiBessonov) on X, posted 2026-06-27
