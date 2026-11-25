---
title: "Derf: Normalization-Free Transformers That Outperform"
publishDate: "2025-12-12"
date: "2025-12-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999321116641497355-1.jpg"
excerpt: "A paper introduces Derf, a point-wise layer that lets norm-free Transformers outperform normalized ones."
---

> Source: [@liuzhuang1234 on X](https://x.com/liuzhuang1234/status/1999321116641497355) · posted 2025-12-12

![Derf: Normalization-Free Transformers That Outperform](/imgs/blogs/x-bookmarks/1999321116641497355-1.jpg)

## Summary

@liuzhuang1234 announces a paper called Stronger Normalization-Free Transformers. It introduces Derf (Dynamic erf), a simple point-wise layer that lets norm-free Transformers work and, they claim, outperform their normalized counterparts.

## Background

Normalization layers like LayerNorm are standard in Transformers. Removing them simplifies the model and may speed up inference. Derf replaces them with an erf-based function applied element-wise.

## What I took away

- Normalization may be replaceable with simple functions.
- The result is the authors' claim.
- See the related bifurcation analysis bookmark.

## Reference

- Original post: [https://x.com/liuzhuang1234/status/1999321116641497355](https://x.com/liuzhuang1234/status/1999321116641497355)
- Author: [@liuzhuang1234](https://x.com/liuzhuang1234) on X, posted 2025-12-12
