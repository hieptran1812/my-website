---
title: "Shampoo Variants Top an Optimizer Batch Size Sweep"
publishDate: "2026-10-10"
date: "2026-10-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "A sweep finds the best optimizer depends on batch size, and Shampoo variants take the top three spots."
---

> Source: [@_arohan_ on X](https://x.com/_arohan_/status/2108977901274275873) · posted 2026-10-10

## Summary

The author notes that the top three optimizers in a new sweep are all Shampoo or its variants, and credits Vineet Gupta as the original author. The quoted post says the best optimizer can change with batch size, and that the best batch size scaling rule depends on settings and cannot be derived from early theories like SDE or convex bounds.

## Background

Shampoo is a second-order-style optimizer that uses matrix preconditioners to adapt step sizes per parameter block, unlike Adam's per-element scaling. Batch size scaling rules tell you how to change learning rate and other settings when you change batch size.

## What I took away

- Optimizer rankings can flip with batch size; test at the scale you will train.
- Do not rely only on theory to set batch size scaling rules.
- Shampoo-family optimizers are worth trying for large-batch training.

## Reference

- Original post: [https://x.com/_arohan_/status/2108977901274275873](https://x.com/_arohan_/status/2108977901274275873)
- Author: [@_arohan_](https://x.com/_arohan_) on X, posted 2026-10-10
