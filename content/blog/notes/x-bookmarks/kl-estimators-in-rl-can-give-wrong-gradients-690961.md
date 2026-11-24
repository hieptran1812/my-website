---
title: "KL Estimators in RL Can Give Wrong Gradients"
publishDate: "2026-07-25"
date: "2026-07-25"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "math", "post-training"]
author: "Hiep Tran"
featured: false
excerpt: "The author notes that standard KL estimators used in RL losses, like k3, can produce wrong gradients for expected KL."
---

> Source: [@HeMuyu0327 on X](https://x.com/HeMuyu0327/status/2080901296337690961) · posted 2026-07-25

## Summary

The author says it is little known that standard **KL estimator** definitions in RL losses produce the wrong gradients for expected KL. He gives the example of the k3 estimator used in the DeepSeek series. While k3 is an unbiased estimate of the KL value, its gradient may not be (text cut off).

## Background

KL divergence measures how far the trained model has drifted from a reference model, and is added as a penalty in RL. Since it cannot be computed exactly, people use sample-based estimators named k1, k2 and k3. An estimator can be unbiased for the value but give biased gradients, which is what training actually uses.

## What I took away

- An unbiased value estimate does not guarantee an unbiased gradient.
- Check which KL estimator and placement your RL code uses.
- Small loss details can change what the model actually optimizes.

## Reference

- Original post: [https://x.com/HeMuyu0327/status/2080901296337690961](https://x.com/HeMuyu0327/status/2080901296337690961)
- Author: [@HeMuyu0327](https://x.com/HeMuyu0327) on X, posted 2026-07-25
