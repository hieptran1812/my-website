---
title: "Day 26 of Learning Inference Engineering and CUDA"
publishDate: "2026-11-23"
date: "2026-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "learning-resources", "gpu-kernels", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124958576636489964-1.jpg"
excerpt: "A learner continues a 100-day inference engineering journey with chapter 1 of CUDA for LLMs about host and device."
---

> Source: [@pavelsimo on X](https://x.com/pavelsimo/status/2124958576636489964) · posted 2026-11-23

![Day 26 of Learning Inference Engineering and CUDA (1)](/imgs/blogs/x-bookmarks/2124958576636489964-1.jpg)

![Day 26 of Learning Inference Engineering and CUDA (2)](/imgs/blogs/x-bookmarks/2124958576636489964-2.jpg)

![Day 26 of Learning Inference Engineering and CUDA (3)](/imgs/blogs/x-bookmarks/2124958576636489964-3.jpg)

![Day 26 of Learning Inference Engineering and CUDA (4)](/imgs/blogs/x-bookmarks/2124958576636489964-4.jpg)

## Summary

The author posts Day 26 of 100 of studying inference engineering, starting chapter 1 of CUDA for LLMs titled when PyTorch just is not enough. They note CUDA programs work in two contexts, the host CPU and the device GPU, with the CPU managing its RAM. The quoted post marks finishing the book Inference Engineering by Philip Kiely.

## Background

In CUDA, the host is the CPU and its memory, and the device is the GPU with its own memory. Data must be moved between them, which is a common source of slowness. Learning this is the base for writing custom GPU kernels.

## What I took away

- Host and device have separate memory in CUDA.
- Public daily learning logs help keep consistency.
- Inference Engineering by Philip Kiely is a recommended starter book.

## Reference

- Original post: [https://x.com/pavelsimo/status/2124958576636489964](https://x.com/pavelsimo/status/2124958576636489964)
- Author: [@pavelsimo](https://x.com/pavelsimo) on X, posted 2026-11-23
