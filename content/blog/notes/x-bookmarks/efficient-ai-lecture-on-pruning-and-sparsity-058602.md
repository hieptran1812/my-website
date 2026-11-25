---
title: "Efficient AI Lecture On Pruning And Sparsity"
publishDate: "2026-03-18"
date: "2026-03-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "learning-resources", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2034108998946058602-1.jpg"
excerpt: "The author shares notes on Efficient AI Lecture 3 about pruning and sparsity, with a Colab demo."
---

> Source: [@ickma2311 on X](https://x.com/ickma2311/status/2034108998946058602) · posted 2026-03-18

![Efficient AI Lecture On Pruning And Sparsity](/imgs/blogs/x-bookmarks/2034108998946058602-1.jpg)

## Summary

The notes say memory can cost more than compute, pruning is an L0 optimization problem, and unstructured pruning is flexible while structured pruning is hardware-friendly. The lecture includes a Colab demo, and the author links their notes.

## Background

Pruning removes weights to shrink a model. Unstructured pruning removes individual weights, which is flexible but hard to speed up on GPUs. Structured pruning removes whole blocks or channels, which hardware can exploit.

## What I took away

- Memory movement is often the real cost, not arithmetic.
- Structured sparsity gives real speedups on hardware.
- Work through the Colab demo to see pruning in practice.

## Reference

- Original post: [https://x.com/ickma2311/status/2034108998946058602](https://x.com/ickma2311/status/2034108998946058602)
- Author: [@ickma2311](https://x.com/ickma2311) on X, posted 2026-03-18
