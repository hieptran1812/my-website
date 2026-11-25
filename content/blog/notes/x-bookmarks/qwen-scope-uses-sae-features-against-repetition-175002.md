---
title: "Qwen Scope Uses SAE Features Against Repetition"
publishDate: "2026-04-30"
date: "2026-04-30"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "interpretability", "reinforcement-learning", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2049788155864175002-1.jpg"
excerpt: "A researcher describes Qwen's first interpretability release using SAE features to find repetition causes and steer for RL negatives."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2049788155864175002) · posted 2026-04-30

![Qwen Scope Uses SAE Features Against Repetition](/imgs/blogs/x-bookmarks/2049788155864175002-1.jpg)

## Summary

The author finds Qwen's first interpretability release, qwen scope, interesting. They use SAE features to identify what causes repetition in outputs, then steer to produce a bad rollout where the model repeats a lot. This gives RL a clear negative signal to learn from. The post is cut off.

## Background

Sparse autoencoders (SAEs) find interpretable features in activations. Steering pushes activations along a feature. Generating bad examples on purpose gives reinforcement learning something to avoid.

## What I took away

- Interpretability can create training signals, not just explanations.
- Steering can manufacture hard negative examples.

## Reference

- Original post: [https://x.com/eliebakouch/status/2049788155864175002](https://x.com/eliebakouch/status/2049788155864175002)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-04-30
