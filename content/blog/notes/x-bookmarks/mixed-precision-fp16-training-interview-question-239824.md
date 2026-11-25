---
title: "Mixed Precision FP16 Training Interview Question"
publishDate: "2026-03-05"
date: "2026-03-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "An ML interview question about training with fp16 while keeping fp32 master weights."
---

> Source: [@prajdabre on X](https://x.com/prajdabre/status/2029693522564239824) · posted 2026-03-05

## Summary

The author poses a basic ML interview question: to speed up training you use fp16, keep fp32 master weights, convert them to fp16 for the forward pass, compute loss, run backward, and map the weights back. The post is cut off before the actual question.

## Background

Mixed precision training uses low precision for speed and fp32 master weights for accurate updates. fp16 has a small range, so small gradients can underflow, which is why loss scaling is often used. The truncated question likely probes such issues.

## What I took away

- Know why fp32 master weights are kept.
- Understand gradient underflow and loss scaling.
- Open the post for the full question.

## Reference

- Original post: [https://x.com/prajdabre/status/2029693522564239824](https://x.com/prajdabre/status/2029693522564239824)
- Author: [@prajdabre](https://x.com/prajdabre) on X, posted 2026-03-05
