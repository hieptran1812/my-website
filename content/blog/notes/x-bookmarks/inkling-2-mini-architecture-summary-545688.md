---
title: "Inkling 2 Mini Architecture Summary"
publishDate: "2026-11-20"
date: "2026-11-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "distillation", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "A quoted post describes Inkling 2 Mini as a 24-layer looped model pruned to 26B total and 5.4B active parameters in NVFP4."
---

> Source: [@SonglinYang4 on X](https://x.com/SonglinYang4/status/2123880674075545688) · posted 2026-11-20

## Summary

The author posts an emoji and quotes the Inkling 2 Mini description. It has 24 layers with the middle 12 looped 3x, pruned to 26B total and 5.4B active parameters, natively in NVFP4, logit-distilled from a big model, small enough to run on your own device, with open weights.

## Background

NVFP4 is a 4-bit floating point format that shrinks memory use. Logit distillation trains a small model to match a larger model's output probabilities. Pruning removes parts of the network.

## What I took away

- Small models can combine looping, pruning, distillation and low precision.
- Active parameters are fewer than total.
- Weights are open per the post.

## Reference

- Original post: [https://x.com/SonglinYang4/status/2123880674075545688](https://x.com/SonglinYang4/status/2123880674075545688)
- Author: [@SonglinYang4](https://x.com/SonglinYang4) on X, posted 2026-11-20
