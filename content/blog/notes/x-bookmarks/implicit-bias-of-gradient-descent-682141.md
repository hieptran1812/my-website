---
title: "Implicit Bias Of Gradient Descent"
publishDate: "2025-11-17"
date: "2025-11-17"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "math", "ai-research", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A post stresses that ML engineers should understand how gradient descent selects one solution among infinitely many."
---

> Source: [@chastronomic on X](https://x.com/chastronomic/status/1990311425575682141) · posted 2025-11-17

## Summary

The author says an ML engineer who does not understand how gradient descent picks one solution out of infinitely many is missing key deep learning theory. This is Concept 10, the implicit bias of gradient descent. The text is cut off after it starts to say modern deep networks are so (large).

## Background

Large networks can fit training data in many ways. Gradient descent, through its starting point and update rule, tends to land on certain solutions, such as simple or low-norm ones. This hidden preference is called implicit bias and helps explain why big models still generalize.

## What I took away

- Training does not just find any solution; the optimizer shapes which one.
- Implicit bias helps explain generalization in overparameterized networks.
- Worth studying as part of deep learning theory.

## Reference

- Original post: [https://x.com/chastronomic/status/1990311425575682141](https://x.com/chastronomic/status/1990311425575682141)
- Author: [@chastronomic](https://x.com/chastronomic) on X, posted 2025-11-17
