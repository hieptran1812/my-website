---
title: "Why RL On One Domain Hurts Another"
publishDate: "2026-06-03"
date: "2026-06-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062171822544527465-1.png"
excerpt: "A paper reports multi-domain RL interference sits in a sparse low-dimensional subspace, fixable with a short refresh."
---

> Source: [@HuggingPapers on X](https://x.com/HuggingPapers/status/2062171822544527465) · posted 2026-06-03

![Why RL On One Domain Hurts Another](/imgs/blogs/x-bookmarks/2062171822544527465-1.png)

## Summary

The post says RL on one domain can crash another. A new local perturbation theory shows the interference is concentrated in a sparse, low-dimensional subspace. The fix is a short refresh that selectively recovers lost skills without collateral damage.

## Background

When you train a model with RL on one task, it can forget others, known as interference. If the damage lives in a small subspace, targeted repairs are possible.

## What I took away

- Multi-domain RL can cause skill loss.
- Interference being low-dimensional suggests cheap fixes.

## Reference

- Original post: [https://x.com/HuggingPapers/status/2062171822544527465](https://x.com/HuggingPapers/status/2062171822544527465)
- Author: [@HuggingPapers](https://x.com/HuggingPapers) on X, posted 2026-06-03
