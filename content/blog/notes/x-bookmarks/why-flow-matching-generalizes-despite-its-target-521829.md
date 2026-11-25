---
title: "Why Flow Matching Generalizes Despite Its Target"
publishDate: "2025-06-18"
date: "2025-06-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "diffusion", "ai-research", "math"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1935246909473521829-1.jpg"
excerpt: "A new paper asks why flow matching generalizes, given that its exact target can only generate training points."
---

> Source: [@mathusmassias on X](https://x.com/mathusmassias/status/1935246909473521829) · posted 2025-06-18

![Why Flow Matching Generalizes Despite Its Target](/imgs/blogs/x-bookmarks/1935246909473521829-1.jpg)

## Summary

Mathurin Massias announces a paper on generalization in flow matching. He points out that the flow matching target you try to learn can only generate training points. The work is with Quentin Bertrand, Anne Gagneux and Remi Emonet.

## Background

Flow matching trains a model to move noise toward data along a learned velocity field. The ideal closed-form target would only reproduce the training set, so if models still create new samples, something about learning must explain it. The paper looks into that gap.

## What I took away

- The exact optimal target of flow matching memorizes the training data.
- Generalization must come from how the network learns an approximation, not from the target itself.
- Read the arXiv paper 2506.03719 for the full argument.

## Reference

- Original post: [https://x.com/mathusmassias/status/1935246909473521829](https://x.com/mathusmassias/status/1935246909473521829)
- Author: [@mathusmassias](https://x.com/mathusmassias) on X, posted 2025-06-18
