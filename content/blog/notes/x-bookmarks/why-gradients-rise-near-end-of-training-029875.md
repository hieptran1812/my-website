---
title: "Why Gradients Rise Near End Of Training"
publishDate: "2026-11-09"
date: "2026-11-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "math", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2119778741950029875-1.png"
excerpt: "Explains the math of gradient growth late in training due to weight decay, learning rate decay and normalization."
---

> Source: [@miyashita_03 on X](https://x.com/miyashita_03/status/2119778741950029875) · posted 2026-11-09

![Why Gradients Rise Near End Of Training (1)](/imgs/blogs/x-bookmarks/2119778741950029875-1.png)

![Why Gradients Rise Near End Of Training (2)](/imgs/blogs/x-bookmarks/2119778741950029875-2.png)

## Summary

The author gives the math of why gradients rapidly increase near the end of training. Weight updates increase weight norm and weight decay reduces it, reaching an equilibrium. The quoted TL;DR says scale-invariant normalization layers shrink weights by c while growing gradients by c, so LR decay with weight decay makes gradients grow.

## Background

Scale invariance means a layer's output does not change if its weights are scaled. Smaller weights then yield larger gradients. When the learning rate decays, the balance with weight decay shifts. This explains a common training curve pattern.

## What I took away

- Gradient norm growth late in training can be expected.
- LR schedule and weight decay interact.
- Do not treat the rise as a bug automatically.

## Reference

- Original post: [https://x.com/miyashita_03/status/2119778741950029875](https://x.com/miyashita_03/status/2119778741950029875)
- Author: [@miyashita_03](https://x.com/miyashita_03) on X, posted 2026-11-09
