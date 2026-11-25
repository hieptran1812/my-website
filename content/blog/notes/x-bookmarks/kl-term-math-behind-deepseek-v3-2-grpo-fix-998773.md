---
title: "KL Term Math Behind DeepSeek v3.2 GRPO Fix"
publishDate: "2025-12-15"
date: "2025-12-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "math"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000420585004998773-1.jpg"
excerpt: "A breakdown of the KL term in GRPO, starting from the k1 and k3 estimators, to understand DeepSeek v3.2's fix."
---

> Source: [@HeMuyu0327 on X](https://x.com/HeMuyu0327/status/2000420585004998773) · posted 2025-12-15

![KL Term Math Behind DeepSeek v3.2 GRPO Fix (1)](/imgs/blogs/x-bookmarks/2000420585004998773-1.jpg)

![KL Term Math Behind DeepSeek v3.2 GRPO Fix (2)](/imgs/blogs/x-bookmarks/2000420585004998773-2.png)

![KL Term Math Behind DeepSeek v3.2 GRPO Fix (3)](/imgs/blogs/x-bookmarks/2000420585004998773-3.jpg)

## Summary

The author wants to understand how DeepSeek v3.2 fixed GRPO, so they work through the math of the KL term. The original GRPO used what John Schulman calls the k3 estimator, which is derived from the true KL term, called the k1 estimator.

## Background

KL divergence measures how far the trained model has drifted from a reference model. It is estimated from samples, and k1 and k3 are two different sample-based estimators with different bias and variance. The choice affects the gradients used in RL training.

## What I took away

- KL penalties keep an RL-trained model close to its reference.
- Different KL estimators (k1, k3) behave differently in practice.
- Check the estimator when reading or reproducing GRPO code.

## Reference

- Original post: [https://x.com/HeMuyu0327/status/2000420585004998773](https://x.com/HeMuyu0327/status/2000420585004998773)
- Author: [@HeMuyu0327](https://x.com/HeMuyu0327) on X, posted 2025-12-15
