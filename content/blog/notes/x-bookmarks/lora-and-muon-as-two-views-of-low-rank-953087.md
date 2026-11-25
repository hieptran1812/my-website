---
title: "LoRA and Muon as Two Views of Low-Rank Gradients"
publishDate: "2025-12-12"
date: "2025-12-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "math", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999490491075953087-1.jpg"
excerpt: "A joke and an explanation that LoRA and Muon both respond to the low-rank structure of gradients in opposite ways."
---

> Source: [@byebyescaling on X](https://x.com/byebyescaling/status/1999490491075953087) · posted 2025-12-12

![LoRA and Muon as Two Views of Low-Rank Gradients](/imgs/blogs/x-bookmarks/1999490491075953087-1.jpg)

## Summary

@byebyescaling writes about a Singular Value Schism. LoRA treats rank collapse of the gradient as a hint to save VRAM, while Muon treats it as degeneracy to be orthogonalized. The quoted post says they are two sides of the same coin: one treats the low-rank gradient structure as a feature, the other as a bug.

## Background

Weight gradients often have a few dominant directions. LoRA exploits this by training small low-rank updates. Muon is an optimizer that orthogonalizes updates so all directions get used. Both start from the same observation.

## What I took away

- Low-rank gradients can be exploited or corrected.
- LoRA saves memory, Muon changes the update geometry.
- Think about which fits your goal.

## Reference

- Original post: [https://x.com/byebyescaling/status/1999490491075953087](https://x.com/byebyescaling/status/1999490491075953087)
- Author: [@byebyescaling](https://x.com/byebyescaling) on X, posted 2025-12-12
