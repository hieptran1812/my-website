---
title: "Error-Free Linear Attention And Long Context Stability"
publishDate: "2025-12-17"
date: "2025-12-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "math", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001218662615925065-1.jpg"
excerpt: "bycloudai asks whether Error-Free Linear Attention removes the Euler discretization error and fixes long-context instability."
---

> Source: [@bycloudai on X](https://x.com/bycloudai/status/2001218662615925065) · posted 2025-12-17

![Error-Free Linear Attention And Long Context Stability (1)](/imgs/blogs/x-bookmarks/2001218662615925065-1.jpg)

![Error-Free Linear Attention And Long Context Stability (2)](/imgs/blogs/x-bookmarks/2001218662615925065-2.jpg)

![Error-Free Linear Attention And Long Context Stability (3)](/imgs/blogs/x-bookmarks/2001218662615925065-3.jpg)

## Summary

The author asks if a free lunch is real. He says the paper shows mathematically that Euler discretization is removed, so forgetting emerges naturally, and wonders whether this removes long context instability of linear attention. The quoted post says the paper introduces Error-Free Linear Attention, a stable linear-time attention it claims is better than other variants and DeltaNet.

## Background

Many linear attention layers can be seen as discretized differential equations that update a memory state. The Euler method is a simple, approximate way to discretize, which adds error. An exact discretization could make the memory update more stable. The claims are from the paper and the author is asking for others' views.

## What I took away

- Linear attention can be viewed as a discretized dynamical system.
- Discretization error may cause instability at long context.
- Treat strong claims as unverified until others reproduce them.

## Reference

- Original post: [https://x.com/bycloudai/status/2001218662615925065](https://x.com/bycloudai/status/2001218662615925065)
- Author: [@bycloudai](https://x.com/bycloudai) on X, posted 2025-12-17
