---
title: "Why Derf Works in Normalization-Free Transformers"
publishDate: "2025-12-14"
date: "2025-12-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "math", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000195657978511388-1.jpg"
excerpt: "A reading of the Derf paper, which replaces LayerNorm with an erf function and claims better generalization."
---

> Source: [@byebyescaling on X](https://x.com/byebyescaling/status/2000195657978511388) · posted 2025-12-14

![Why Derf Works in Normalization-Free Transformers (1)](/imgs/blogs/x-bookmarks/2000195657978511388-1.jpg)

![Why Derf Works in Normalization-Free Transformers (2)](/imgs/blogs/x-bookmarks/2000195657978511388-2.jpg)

## Summary

@byebyescaling read the Derf paper by @liuzhuang1234 and colleagues and offers a bifurcation analysis of why it works. The central claim is that replacing LayerNorm with Derf(x) = erf(alpha x + s) improves generalization. The quoted post introduces Derf as a point-wise layer that lets norm-free Transformers outperform normalized ones.

## Background

LayerNorm normalizes activations inside a Transformer and needs statistics across features. Derf is a simple per-element function with no such statistics. Bifurcation analysis studies how a system's behavior changes as parameters change, here used to explain why training is stable.

## What I took away

- Normalization layers may be replaceable by simple point-wise functions.
- Point-wise layers avoid cross-feature statistics, which can simplify kernels.
- Results are the authors' claims; check the paper's experiments.

## Reference

- Original post: [https://x.com/byebyescaling/status/2000195657978511388](https://x.com/byebyescaling/status/2000195657978511388)
- Author: [@byebyescaling](https://x.com/byebyescaling) on X, posted 2025-12-14
