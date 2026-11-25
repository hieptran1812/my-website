---
title: "Muon Versus Adam And SinkGD"
publishDate: "2026-06-29"
date: "2026-06-29"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2071605967737721044-1.png"
excerpt: "A post says in tests Muon beat Adam and Sinkhorn GD across many linear layers."
---

> Source: [@murage_kibicho on X](https://x.com/murage_kibicho/status/2071605967737721044) · posted 2026-06-29

![Muon Versus Adam And SinkGD](/imgs/blogs/x-bookmarks/2071605967737721044-1.png)

## Summary

@murage_kibicho says they compared many linear layers and Muon beat Adam and Sinkhorn GD, and shows a loss plot. The quoted post claims big labs use SGD with a spectral norm regularizer to save optimizer memory.

## Background

Adam stores mean and variance per parameter. Muon orthogonalizes updates. Saving optimizer memory permits larger batches.

## What I took away

- Optimizer choice affects memory and batch size.
- The lab practice claim is a rumor.

## Reference

- Original post: [https://x.com/murage_kibicho/status/2071605967737721044](https://x.com/murage_kibicho/status/2071605967737721044)
- Author: [@murage_kibicho](https://x.com/murage_kibicho) on X, posted 2026-06-29
