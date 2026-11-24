---
title: "RMSNorm Versus LayerNorm at the Kernel Level"
publishDate: "2026-07-12"
date: "2026-07-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-architecture"]
author: "Hiep Tran"
featured: false
excerpt: "An argument that research teams should learn how model code lowers to GPU kernels, using RMSNorm versus LayerNorm."
---

> Source: [@waterloo_intern on X](https://x.com/waterloo_intern/status/2076226016260063728) · posted 2026-07-12

## Summary

The author argues every research team should learn how its modeling code lowers to GPU kernels. His example: the industry assumes RMSNorm is cheaper than LayerNorm because it skips mean-centering and looks simpler. He says this assumption is not true.

## Background

LayerNorm subtracts the mean and divides by the standard deviation; RMSNorm only divides by the root mean square. Both are memory-bound: their speed depends on reading and writing data, not on arithmetic. So removing a few math steps may not make them faster.

## What I took away

- Simpler math does not mean a faster kernel when the op is memory-bound.
- Profile before assuming an architecture change saves time.
- See the reply from @cHHillee for a more nuanced view.

## Reference

- Original post: [https://x.com/waterloo_intern/status/2076226016260063728](https://x.com/waterloo_intern/status/2076226016260063728)
- Author: [@waterloo_intern](https://x.com/waterloo_intern) on X, posted 2026-07-12
