---
title: "Compositional Muon Optimizer For Attention Circuits"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "pretraining", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062964127245451523-1.png"
excerpt: "A reply explains Compositional Muon, which preconditions query gradients using the key weights in attention."
---

> Source: [@varunneal on X](https://x.com/varunneal/status/2062964127245451523) · posted 2026-06-05

![Compositional Muon Optimizer For Attention Circuits](/imgs/blogs/x-bookmarks/2062964127245451523-1.png)

## Summary

The author reacts to Tilde Research's Compositional Muon, an optimizer that extends Muon from single matrices to composed transformer circuits like QK^T and OV. Their understanding is that query gradients are preconditioned by the eigenbasis of the key weights. This whitens how much a query update is amplified by the key matrix in the downstream loss.

## Background

Muon is an optimizer that treats each weight matrix as a unit when it updates it. In attention, the loss depends on products such as Q times K transposed, so two matrices act together. Updating them separately can over- or under-step, which is what this method tries to correct.

## What I took away

- Attention loss depends on products of matrices, not on single weights.
- Optimizer trust regions may work better when drawn around those compositions.
- Check the quoted Tilde Research post for the actual method.

## Reference

- Original post: [https://x.com/varunneal/status/2062964127245451523](https://x.com/varunneal/status/2062964127245451523)
- Author: [@varunneal](https://x.com/varunneal) on X, posted 2026-06-05
