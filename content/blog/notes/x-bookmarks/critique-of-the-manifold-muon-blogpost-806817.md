---
title: "Critique Of The Manifold Muon Blogpost"
publishDate: "2025-11-19"
date: "2025-11-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "math", "ai-research"]
author: "Hiep Tran"
featured: false
excerpt: "Jason Lee says the manifold muon blogpost makes no sense and questions why the gradient is projected onto the Stiefel tangent space."
---

> Source: [@jasondeanlee on X](https://x.com/jasondeanlee/status/1991047111467806817) · posted 2025-11-19

## Summary

The author says the manifold muon blogpost makes zero sense. They ask why you would project the gradient onto the Stiefel tangent before the spectral norm or polar step, and say the only advantage they see is staying approximately on the Stiefel manifold up to second order in the step size. The text is cut off.

## Background

Muon is an optimizer that orthogonalizes gradient updates for matrix weights. The Stiefel manifold is the set of matrices with orthonormal columns, and manifold muon tries to keep weights on it. The author is questioning whether the extra projection adds real benefit.

## What I took away

- Ask what a design step actually buys before adding it.
- Optimizer ideas need careful math and evidence.
- This is one researcher's critique, so read the original blog too.

## Reference

- Original post: [https://x.com/jasondeanlee/status/1991047111467806817](https://x.com/jasondeanlee/status/1991047111467806817)
- Author: [@jasondeanlee](https://x.com/jasondeanlee) on X, posted 2025-11-19
