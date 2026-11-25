---
title: "Autokernel Speeds Up GPU Kernels Automatically"
publishDate: "2026-03-21"
date: "2026-03-21"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "ai-research", "llm-inference"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2035385195809960133-1.jpg"
excerpt: "A thread entry quotes autokernel, an open-source autoresearch loop that profiles a PyTorch model and writes Triton kernel replacements."
---

> Source: [@zhengyaojiang on X](https://x.com/zhengyaojiang/status/2035385195809960133) · posted 2026-03-21

![Autokernel Speeds Up GPU Kernels Automatically](/imgs/blogs/x-bookmarks/2035385195809960133-1.jpg)

## Summary

This is item 4 of 6 in a thread on speeding up GPU kernels. The quoted post open-sources autokernel: give it any PyTorch model, it profiles it, finds bottleneck kernels, writes Triton replacements and runs experiments overnight, editing one file, benchmarking, and keeping or reverting changes in a loop.

## Background

A GPU kernel is a small program that runs on the GPU; Triton is a Python-like language for writing them. Optimizing kernels by hand is slow, so an automated loop with a benchmark as the metric is a natural fit.

## What I took away

- Kernel tuning fits autoresearch since speed is easy to measure.
- Always check correctness of generated kernels, not just speed.
- Keep or revert based on benchmark is a simple, safe loop.

## Reference

- Original post: [https://x.com/zhengyaojiang/status/2035385195809960133](https://x.com/zhengyaojiang/status/2035385195809960133)
- Author: [@zhengyaojiang](https://x.com/zhengyaojiang) on X, posted 2026-03-21
