---
title: "Content-Based Sparse Attention History"
publishDate: "2026-06-12"
date: "2026-06-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "llm-inference", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2065239986740596876-1.png"
excerpt: "A post noting MiniMax pushing content-based sparse attention, with a recap of 2020 efficient attention variants."
---

> Source: [@aurko79 on X](https://x.com/aurko79/status/2065239986740596876) · posted 2026-06-12

![Content-Based Sparse Attention History (1)](/imgs/blogs/x-bookmarks/2065239986740596876-1.png)

![Content-Based Sparse Attention History (2)](/imgs/blogs/x-bookmarks/2065239986740596876-2.jpg)

## Summary

@aurko79 is glad MiniMax and others are advancing content-based sparse attention. He recalls that in 2020 many efficient attention variants appeared, such as local window, fixed sparsity, strided and recurrence. The quote announces the open-source MSA kernel library and M3 weights soon.

## Background

Standard attention cost grows quadratically with context. Sparse attention skips most pairs. Content-based methods choose which tokens to attend based on the data rather than fixed patterns.

## What I took away

- Content-based selection is a newer flavor of an older idea.
- Kernels matter as much as the algorithm.

## Reference

- Original post: [https://x.com/aurko79/status/2065239986740596876](https://x.com/aurko79/status/2065239986740596876)
- Author: [@aurko79](https://x.com/aurko79) on X, posted 2026-06-12
