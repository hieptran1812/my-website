---
title: "Argument That Softmax Attention Uses Wrong Norm"
publishDate: "2025-12-24"
date: "2025-12-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research", "math"]
author: "Hiep Tran"
featured: false
excerpt: "A post and article argue that softmax attention normalizes by the L1 norm when it should use the L2 norm."
---

> Source: [@Clashluke on X](https://x.com/Clashluke/status/2003873670393991654) · posted 2025-12-24

## Summary

Clashluke says attention is broken. The linked article is titled Attention Normalizes the Wrong Norm and argues that softmax constrains the L1 norm of the weights to 1, but should constrain the L2 norm.

## Background

Softmax makes attention weights positive and sum to 1, which is an L1 constraint. The article argues that an L2 constraint would be better. This is a bold claim by the author, so look for experiments before accepting it.

## What I took away

- Even standard parts of transformers can be questioned.
- Check whether the claim is backed by experiments.

## Reference

- Original post: [https://x.com/Clashluke/status/2003873670393991654](https://x.com/Clashluke/status/2003873670393991654)
- Author: [@Clashluke](https://x.com/Clashluke) on X, posted 2025-12-24
