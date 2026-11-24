---
title: "Day 4 of Learning Blackwell GPU Kernels"
publishDate: "2026-09-12"
date: "2026-09-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "hardware", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "A learning log on Blackwell kernels covering instruction sets, TMA descriptors, layouts, tcgen05 and tensor memory."
---

> Source: [@jino_rohit on X](https://x.com/jino_rohit/status/2098656920496419231) · posted 2026-09-12

## Summary

On day 4 of 30, the author studied instruction sets, TMA descriptors and layouts, and found them confusing without visualization. They also looked at a simple TMA multicast kernel. In the quoted day 3 post, they read about tcgen05 instructions and tensor memory, where MMA results are stored on dedicated hardware, removing WGMMA's dependence on registers.

## Background

Blackwell is NVIDIA's GPU generation after Hopper. TMA (Tensor Memory Accelerator) moves tiles of data between global and shared memory; multicast sends the same tile to several cores. Tensor memory is a new on-chip store for matrix multiply results, which frees up registers.

## What I took away

- Visualize memory layouts when learning TMA and tiling.
- Blackwell changes where matmul results live, so Hopper habits need updating.
- A daily learning log is a good way to stay consistent.

## Reference

- Original post: [https://x.com/jino_rohit/status/2098656920496419231](https://x.com/jino_rohit/status/2098656920496419231)
- Author: [@jino_rohit](https://x.com/jino_rohit) on X, posted 2026-09-12
