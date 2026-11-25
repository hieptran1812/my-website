---
title: "Hand-Drawn Notes On RNN Gradients"
publishDate: "2025-12-13"
date: "2025-12-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999838921887236194-1.jpg"
excerpt: "A recommendation of a learner's hand-drawn notes explaining why gradients vanish or explode in recurrent neural networks."
---

> Source: [@Clashluke on X](https://x.com/Clashluke/status/1999838921887236194) · posted 2025-12-13

![Hand-Drawn Notes On RNN Gradients (1)](/imgs/blogs/x-bookmarks/1999838921887236194-1.jpg)

![Hand-Drawn Notes On RNN Gradients (2)](/imgs/blogs/x-bookmarks/1999838921887236194-2.jpg)

## Summary

@Clashluke tells readers to check out the notes of @ickma2311, saying they will blow your mind. The quoted post is by that author, who says they drew the figures themselves to build intuition about why gradients vanish or explode in RNNs.

## Background

An RNN (recurrent neural network) reads a sequence step by step and reuses the same weights each step. When gradients are passed back through many steps, they get multiplied again and again, so they can shrink toward zero (vanish) or grow very large (explode). This is a classic reason plain RNNs are hard to train on long sequences.

## What I took away

- Drawing the figures yourself is a good way to build intuition for a concept.
- Vanishing and exploding gradients come from repeated multiplication across time steps.
- Personal learning notes can be worth sharing and reading.

## Reference

- Original post: [https://x.com/Clashluke/status/1999838921887236194](https://x.com/Clashluke/status/1999838921887236194)
- Author: [@Clashluke](https://x.com/Clashluke) on X, posted 2025-12-13
