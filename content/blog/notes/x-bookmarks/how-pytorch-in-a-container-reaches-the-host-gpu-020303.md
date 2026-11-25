---
title: "How PyTorch In A Container Reaches The Host GPU"
publishDate: "2025-12-01"
date: "2025-12-01"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "tools", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1995474470820020303-1.png"
excerpt: "A thread explaining how a container without GPU drivers can still use the host's GPU through PyTorch."
---

> Source: [@danielepolencic on X](https://x.com/danielepolencic/status/1995474470820020303) · posted 2025-12-01

![How PyTorch In A Container Reaches The Host GPU](/imgs/blogs/x-bookmarks/1995474470820020303-1.png)

## Summary

The author poses the question: if a container has no GPU drivers installed, how does PyTorch inside it use the host GPU? They say they will explain it in a thread.

## Background

Containers share the host's kernel, so the GPU driver lives on the host. Tools like the NVIDIA container toolkit expose the device files and driver libraries into the container. This is why a CUDA container works without installing the kernel driver.

## What I took away

- The kernel driver stays on the host; the container gets access to it.
- Learn the NVIDIA container toolkit for GPU Docker setups.
- Mismatched driver and CUDA versions are a common source of errors.

## Reference

- Original post: [https://x.com/danielepolencic/status/1995474470820020303](https://x.com/danielepolencic/status/1995474470820020303)
- Author: [@danielepolencic](https://x.com/danielepolencic) on X, posted 2025-12-01
