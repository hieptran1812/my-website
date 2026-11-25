---
title: "A Recipe For JEPA With SIGReg"
publishDate: "2025-11-22"
date: "2025-11-22"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "pretraining", "world-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1992230768702328912-1.jpg"
excerpt: "A summary of LeJEPA: use isotropic Gaussian embeddings with SIGReg regularization to train JEPA models."
---

> Source: [@TheTuringPost on X](https://x.com/TheTuringPost/status/1992230768702328912) · posted 2025-11-22

![A Recipe For JEPA With SIGReg](/imgs/blogs/x-bookmarks/1992230768702328912-1.jpg)

## Summary

The post lists a recipe for JEPA (Joint-Embedding Predictive Architecture). The optimal embedding distribution is an isotropic Gaussian, SIGReg regularization is used to reach it, and LeJEPA is said to prove the method works in practice. It refers to Yann LeCun's latest work.

## Background

JEPA models learn by predicting the embedding of one view from another instead of reconstructing pixels or words. A risk is collapse, where all embeddings become the same. Forcing embeddings toward a Gaussian shape is a way to prevent that without many heuristics.

## What I took away

- JEPA predicts in embedding space rather than input space.
- Regularizing embeddings toward a Gaussian can avoid collapse.
- Check the LeJEPA paper for the exact claims.

## Reference

- Original post: [https://x.com/TheTuringPost/status/1992230768702328912](https://x.com/TheTuringPost/status/1992230768702328912)
- Author: [@TheTuringPost](https://x.com/TheTuringPost) on X, posted 2025-11-22
