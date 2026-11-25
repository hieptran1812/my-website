---
title: "Portable Triton Attention Backend Matches FlashAttention 3"
publishDate: "2026-03-06"
date: "2026-03-06"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2029923221445755040-1.jpg"
excerpt: "Red Hat AI says an 800-line Triton attention backend in vLLM matches FlashAttention 3 on H100 and runs on several GPU vendors."
---

> Source: [@RedHat_AI on X](https://x.com/RedHat_AI/status/2029923221445755040) · posted 2026-03-06

![Portable Triton Attention Backend Matches FlashAttention 3 (1)](/imgs/blogs/x-bookmarks/2029923221445755040-1.jpg)

![Portable Triton Attention Backend Matches FlashAttention 3 (2)](/imgs/blogs/x-bookmarks/2029923221445755040-2.jpg)

## Summary

The post says FlashAttention 3 has about 70,000 lines of code while the vLLM Triton attention backend has about 800. On H100 it reaches 100.7% of FlashAttention 3 performance, and on AMD MI300 it gives a 5.8x speedup over earlier implementations, with the same source on NVIDIA, AMD and Intel GPUs. IBM Research, Red Hat and AMD worked on it. These are the authors' claims.

## Background

Triton is a Python-based language for writing GPU kernels. Maintaining a separate attention kernel per GPU platform is costly. A single portable kernel that stays fast reduces that burden.

## What I took away

- One portable kernel can replace many hand-tuned ones.
- Code size and speed are separate wins; verify on your own hardware.
- Triton is worth learning for cross-vendor GPU work.

## Reference

- Original post: [https://x.com/RedHat_AI/status/2029923221445755040](https://x.com/RedHat_AI/status/2029923221445755040)
- Author: [@RedHat_AI](https://x.com/RedHat_AI) on X, posted 2026-03-06
