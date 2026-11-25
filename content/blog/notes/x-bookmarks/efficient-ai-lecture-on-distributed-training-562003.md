---
title: "Efficient AI Lecture on Distributed Training"
publishDate: "2026-06-10"
date: "2026-06-10"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "gpu-kernels", "pretraining"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064536708180562003-1.jpg"
excerpt: "Notes on a lecture about how attention parallelism flows across GPUs under tensor parallelism."
---

> Source: [@ickma2311 on X](https://x.com/ickma2311/status/2064536708180562003) · posted 2026-06-10

![Efficient AI Lecture on Distributed Training](/imgs/blogs/x-bookmarks/2064536708180562003-1.jpg)

## Summary

@ickma2311 shares notes from Efficient AI Lecture 19 on distributed training. It clarified how self-attention parallelism flows, starting from Q, K, V projections and tensor parallelism. The text is cut off.

## Background

Tensor parallelism splits matrices across GPUs, often by attention heads. This needs communication between GPUs.

## What I took away

- Heads are a natural split for parallelism.
- Study the lecture series.

## Reference

- Original post: [https://x.com/ickma2311/status/2064536708180562003](https://x.com/ickma2311/status/2064536708180562003)
- Author: [@ickma2311](https://x.com/ickma2311) on X, posted 2026-06-10
