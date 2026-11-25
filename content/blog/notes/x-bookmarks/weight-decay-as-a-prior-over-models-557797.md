---
title: "Weight Decay As A Prior Over Models"
publishDate: "2025-11-26"
date: "2025-11-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A post argues weight decay is a prior over the model, not just overfitting reduction."
---

> Source: [@chastronomic on X](https://x.com/chastronomic/status/1993668432781557797) · posted 2025-11-26

## Summary

The author says that thinking weight decay is only about reducing overfitting misses the deeper mechanism. In Concept 19 of their series they claim weight decay is a prior over your model, not just shrinking weights, and changes what family of functions your model prefers. The text is cut off.

## Background

Weight decay penalizes large weights during training. In Bayesian terms, L2 weight decay is equivalent to assuming weights come from a Gaussian prior centered at zero. So it encodes a preference for simpler functions.

## What I took away

- Weight decay equals a Gaussian prior on the weights.
- Regularization expresses assumptions about what solutions are good.

## Reference

- Original post: [https://x.com/chastronomic/status/1993668432781557797](https://x.com/chastronomic/status/1993668432781557797)
- Author: [@chastronomic](https://x.com/chastronomic) on X, posted 2025-11-26
