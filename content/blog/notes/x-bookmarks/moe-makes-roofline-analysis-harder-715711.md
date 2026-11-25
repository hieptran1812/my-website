---
title: "MoE Makes Roofline Analysis Harder"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "moe", "hardware", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071513644177715711-1.jpg"
excerpt: "A post says MoE complicates roofline analysis and quotes a correction on matmul intensity."
---

> Source: [@norpadon on X](https://x.com/norpadon/status/2071513644177715711) · posted 2026-06-29

![MoE Makes Roofline Analysis Harder](/imgs/blogs/x-bookmarks/2071513644177715711-1.jpg)

## Summary

@norpadon says MoE models are painful because they make roofline analysis harder. The quoted post says in the JAX scaling book num_tokens < matrix_size is assumed, but now the opposite holds, so intensity is DF/(D+F), yet matmul stays compute-bound.

## Background

Roofline analysis shows whether work is limited by compute or memory. Arithmetic intensity is flops per byte.

## What I took away

- Check assumptions in scaling book formulas.
- MoE changes memory patterns.

## Reference

- Original post: [https://x.com/norpadon/status/2071513644177715711](https://x.com/norpadon/status/2071513644177715711)
- Author: [@norpadon](https://x.com/norpadon) on X, posted 2026-06-29
