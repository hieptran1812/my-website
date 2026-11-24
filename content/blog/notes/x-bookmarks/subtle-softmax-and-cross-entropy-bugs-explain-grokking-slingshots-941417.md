---
title: "Subtle Softmax and Cross-Entropy Bugs Explain Grokking Slingshots"
publishDate: "2026-10-23"
date: "2026-10-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "math", "pretraining"]
author: "Hiep Tran"
featured: false
excerpt: "A researcher says a new result about bugs in softmax and cross-entropy implementations explains slingshots seen in their earlier paper."
---

> Source: [@zhaisf on X](https://x.com/zhaisf/status/2113758728784941417) · posted 2026-10-23

## Summary

@zhaisf says an early version of this work explained what his team saw in their Slingshot paper. The quoted author claims all existing softmax and cross-entropy implementations, including JAX, PyTorch and textbooks, have two subtle bugs, and that a simple rewrite fixes both and eliminates slingshots during grokking.

## Background

Softmax turns raw scores into probabilities, and cross-entropy is the standard loss for classification and language modeling. Grokking is when a model suddenly generalizes long after it memorized the training data. Slingshots are sudden spikes or jumps in training loss linked to numerical behavior late in training.

## What I took away

- Numerical details in basic functions can cause strange training dynamics.
- The claim that all implementations are buggy is the author's; check the write-up.
- Training instabilities may be worth investigating at the level of loss computation.

## Reference

- Original post: [https://x.com/zhaisf/status/2113758728784941417](https://x.com/zhaisf/status/2113758728784941417)
- Author: [@zhaisf](https://x.com/zhaisf) on X, posted 2026-10-23
