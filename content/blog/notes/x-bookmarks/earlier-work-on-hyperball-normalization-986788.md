---
title: "Earlier Work On Hyperball Normalization"
publishDate: "2026-01-23"
date: "2026-01-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2014540515962986788-1.png"
excerpt: "A researcher notes that hyperball normalization for CNNs was already studied in 2018 and welcomes the new Hyperball optimizer wrapper."
---

> Source: [@Besteuler on X](https://x.com/Besteuler/status/2014540515962986788) · posted 2026-01-23

![Earlier Work On Hyperball Normalization (1)](/imgs/blogs/x-bookmarks/2014540515962986788-1.png)

![Earlier Work On Hyperball Normalization (2)](/imgs/blogs/x-bookmarks/2014540515962986788-2.png)

![Earlier Work On Hyperball Normalization (3)](/imgs/blogs/x-bookmarks/2014540515962986788-3.jpg)

## Summary

@Besteuler says doing proper normalization is important for stable neural network training. They point out that their 2018 paper considered hyperball normalization for convolutional networks and proposed more ideas. The quoted post introduces Hyperball, an optimizer wrapper that keeps weight and update norms constant and controls the angular step size, with claimed speedups and good hyperparameter transfer.

## Background

Normalization controls the scale of weights or activations so training stays stable. Keeping the weight norm constant means only the direction changes, so the step size becomes an angle. The reply is about credit and prior art as well as the idea.

## What I took away

- Fixing weight norms makes step size easier to reason about.
- Check prior work before claiming novelty.
- Hyperparameter transfer across scales saves tuning cost.

## Reference

- Original post: [https://x.com/Besteuler/status/2014540515962986788](https://x.com/Besteuler/status/2014540515962986788)
- Author: [@Besteuler](https://x.com/Besteuler) on X, posted 2026-01-23
