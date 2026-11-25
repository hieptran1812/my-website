---
title: "FlexAttention Gets a FlashAttention-4 Backend"
publishDate: "2026-03-05"
date: "2026-03-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2029639428990456233-1.jpg"
excerpt: "A reaction to FlexAttention adding a FlashAttention-4 backend for faster custom attention variants."
---

> Source: [@cloneofsimo on X](https://x.com/cloneofsimo/status/2029639428990456233) · posted 2026-03-05

![FlexAttention Gets a FlashAttention-4 Backend](/imgs/blogs/x-bookmarks/2029639428990456233-1.jpg)

## Summary

The author reacts with surprise to a post saying FlexAttention now has a FlashAttention-4 backend. The quoted post says FlexAttention let researchers prototype custom attention variants, with 1000+ repos adopting it, but users hit a performance ceiling that the new backend addresses.

## Background

FlexAttention lets you write custom attention patterns in a few lines of PyTorch and compiles them to fast kernels. A faster backend means custom variants run closer to hand-tuned speed.

## What I took away

- Prototype attention variants with FlexAttention.
- Backends can improve without changing your code.
- Check the post for the performance details.

## Reference

- Original post: [https://x.com/cloneofsimo/status/2029639428990456233](https://x.com/cloneofsimo/status/2029639428990456233)
- Author: [@cloneofsimo](https://x.com/cloneofsimo) on X, posted 2026-03-05
