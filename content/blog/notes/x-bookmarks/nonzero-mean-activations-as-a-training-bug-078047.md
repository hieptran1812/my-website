---
title: "Nonzero Mean Activations as a Training Bug"
publishDate: "2026-10-31"
date: "2026-10-31"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A claim that common activations have a nonzero mean that causes loss spikes, and that fixing it helps training."
---

> Source: [@yacinelearning on X](https://x.com/yacinelearning/status/2116513577813078047) · posted 2026-10-31

## Summary

The quoted post claims a bug in common activation functions like ReLU, SiLU and GeLU: their post-activation mean is not zero. It says this leads to training pathologies such as loss spikes and representation overshooting, and that removing it helps. @yacinelearning wonders if any part of the deep learning stack is free of bugs.

## Background

An activation function adds nonlinearity after a linear layer. ReLU-like functions output mostly positive values, so their average is above zero, which can shift the inputs to later layers. Normalization layers partly address this.

## What I took away

- Small statistical biases in layers can affect training stability.
- Loss spikes may have architectural causes worth checking.
- Treat this as a claim and look for independent results.

## Reference

- Original post: [https://x.com/yacinelearning/status/2116513577813078047](https://x.com/yacinelearning/status/2116513577813078047)
- Author: [@yacinelearning](https://x.com/yacinelearning) on X, posted 2026-10-31
