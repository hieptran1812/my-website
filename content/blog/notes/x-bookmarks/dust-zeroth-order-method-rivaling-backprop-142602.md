---
title: "Dust Zeroth-Order Method Rivaling Backprop"
publishDate: "2026-10-05"
date: "2026-10-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher reacts to Dust, a zeroth-order method claimed to pretrain transformers close to backprop quality."
---

> Source: [@wen_kaiyue on X](https://x.com/wen_kaiyue/status/2107232316041142602) · posted 2026-10-05

## Summary

The author finds the idea cool and says its activation optimization reminds them of an earlier arXiv paper, wondering how optimizing activations with gradient descent would work. The quoted post introduces Dust, described as the first zeroth-order method to pretrain transformers that approaches and sometimes exceeds backprop given large compute.

## Background

Backpropagation computes exact gradients by passing errors backward through the network. Zeroth-order methods estimate the update direction only from function values, for example by trying random perturbations, which needs no backward pass but usually far more compute.

## What I took away

- Alternatives to backprop could matter for hardware that cannot store activations.
- The claim of exceeding backprop comes with large compute; note the cost.
- Follow the comparison to earlier activation-optimization work.

## Reference

- Original post: [https://x.com/wen_kaiyue/status/2107232316041142602](https://x.com/wen_kaiyue/status/2107232316041142602)
- Author: [@wen_kaiyue](https://x.com/wen_kaiyue) on X, posted 2026-10-05
