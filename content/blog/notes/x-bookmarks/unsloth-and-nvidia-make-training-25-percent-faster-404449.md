---
title: "Unsloth And NVIDIA Make Training 25 Percent Faster"
publishDate: "2026-05-06"
date: "2026-05-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "gpu-kernels", "moe"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2052097306569404449-1.jpg"
excerpt: "Unsloth says working with NVIDIA made LLM training about 25% faster through three optimizations."
---

> Source: [@danielhanchen on X](https://x.com/danielhanchen/status/2052097306569404449) · posted 2026-05-06

![Unsloth And NVIDIA Make Training 25 Percent Faster](/imgs/blogs/x-bookmarks/2052097306569404449-1.jpg)

## Summary

Unsloth says it collaborated with NVIDIA to make LLM training about 25% faster and wrote a guide. The quoted post lists three optimizations: packed-sequence metadata caching, double-buffered checkpoint reloads, and faster MoE routing.

## Background

Packed sequences join short samples to avoid padding waste. Double-buffering overlaps loading with compute. MoE routing picks which experts process each token. The 25% figure is the authors' claim.

## What I took away

- Training speedups often come from removing overhead, not new math.
- Read the guide to see which ideas apply to your setup.

## Reference

- Original post: [https://x.com/danielhanchen/status/2052097306569404449](https://x.com/danielhanchen/status/2052097306569404449)
- Author: [@danielhanchen](https://x.com/danielhanchen) on X, posted 2026-05-06
