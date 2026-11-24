---
title: "Symmetry Explains Gradients in Softmax and Scale Invariance"
publishDate: "2026-11-07"
date: "2026-11-07"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "math", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "The gradient properties of softmax shift invariance and network scale invariance can be derived with the same symmetry-based steps."
---

> Source: [@miyashita_03 on X](https://x.com/miyashita_03/status/2119055712185954651) · posted 2026-11-07

## Summary

@miyashita_03 notes that two earlier threads of his, one on the shift invariance of softmax and one on the scale invariance of a network, lead to gradient properties derived in exactly the same way. Both start from a symmetry, a transformation of inputs or weights that leaves the output unchanged.

## Background

Softmax is shift invariant: adding the same constant to all inputs gives the same output. Some networks are scale invariant: scaling certain weights (for example before a normalization layer) does not change the output. When the output does not change along some direction, the gradient along that direction is zero, which constrains how training moves the parameters.

## What I took away

- Look for symmetries in a model; they often imply simple facts about gradients.
- Softmax gradients sum to zero across inputs because of shift invariance.
- Scale invariance means gradients are orthogonal to the weights, which affects how weight norms evolve.

## Reference

- Original post: [https://x.com/miyashita_03/status/2119055712185954651](https://x.com/miyashita_03/status/2119055712185954651)
- Author: [@miyashita_03](https://x.com/miyashita_03) on X, posted 2026-11-07
