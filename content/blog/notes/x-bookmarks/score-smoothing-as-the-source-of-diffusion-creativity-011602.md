---
title: "Score Smoothing as the Source of Diffusion Creativity"
publishDate: "2026-07-16"
date: "2026-07-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "math", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "Research argues diffusion models generate novel data because network regularization smooths the learned score function."
---

> Source: [@WangBinxu on X](https://x.com/WangBinxu/status/2077574726487011602) · posted 2026-07-16

## Summary

The quoted post asks what drives a diffusion model's ability to generate new data instead of memorizing. It argues this is a mathematical consequence of neural network regularization causing score smoothing and interpolation, with a blog post explaining more. The author praises the intuition that score smoothing and direct density smoothing have different effects, which may explain generalization along the data manifold.

## Background

Diffusion models learn a score function: the direction that moves a noisy sample toward real data. A perfectly fit score would only reproduce training examples. Smoothing it, which happens naturally from network limits and regularization, fills in between examples and creates new samples.

## What I took away

- Imperfect fitting can be the reason models generalize rather than memorize.
- Where you apply smoothing, score versus density, changes what the model generates.
- Read the blog for the math behind the claim.

## Reference

- Original post: [https://x.com/WangBinxu/status/2077574726487011602](https://x.com/WangBinxu/status/2077574726487011602)
- Author: [@WangBinxu](https://x.com/WangBinxu) on X, posted 2026-07-16
