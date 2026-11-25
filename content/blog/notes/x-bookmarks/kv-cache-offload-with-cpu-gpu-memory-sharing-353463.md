---
title: "KV Cache Offload With CPU-GPU Memory Sharing"
publishDate: "2025-11-19"
date: "2025-11-19"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1991262100023353463-1.jpg"
excerpt: "PyTorch shares an NVIDIA blog on accelerating large-scale LLM inference and KV cache offload with CPU-GPU memory sharing."
---

> Source: [@PyTorch on X](https://x.com/PyTorch/status/1991262100023353463) · posted 2025-11-19

![KV Cache Offload With CPU-GPU Memory Sharing](/imgs/blogs/x-bookmarks/1991262100023353463-1.jpg)

## Summary

The PyTorch account points to an NVIDIA developer blog on optimizing applications that use large models with limited memory. It covers accelerating large-scale LLM inference and KV cache offload using CPU-GPU memory sharing.

## Background

The KV cache stores past attention data and grows with context length, often exceeding GPU memory. Offloading moves part of it to CPU memory so longer contexts or more users fit. Shared memory between CPU and GPU makes the move cheaper.

## What I took away

- KV cache size is a key limit on long-context serving.
- Offloading trades some speed for capacity.
- Read the blog for specifics.

## Reference

- Original post: [https://x.com/PyTorch/status/1991262100023353463](https://x.com/PyTorch/status/1991262100023353463)
- Author: [@PyTorch](https://x.com/PyTorch) on X, posted 2025-11-19
