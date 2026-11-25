---
title: "Agent Builds a Zig CUDA Inference Engine Alone"
publishDate: "2026-11-23"
date: "2026-11-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-inference", "gpu-kernels", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2124975715141689818-1.jpg"
excerpt: "A post claims an agent built a zero-dependency Zig and CUDA inference engine on an RTX 5090 with large speed gains over llama.cpp."
---

> Source: [@tensorfish on X](https://x.com/tensorfish/status/2124975715141689818) · posted 2026-11-23

![Agent Builds a Zig CUDA Inference Engine Alone](/imgs/blogs/x-bookmarks/2124975715141689818-1.jpg)

## Summary

The author describes pointing Gemini 4 Neon at an inference optimization skill and an RTX 5090, then leaving it alone. They claim it produced a zero-dependency Zig and sm_120a CUDA inference engine with 370 to 440 tok/s single-user, which they call 4x llama.cpp, and 1M context in 22.7 GB of VRAM.

## Background

Inference engines run trained models to generate text. llama.cpp is a popular open engine. Autonomous agents writing optimized GPU code is an emerging idea, but these numbers are the author's claim and depend on the exact setup.

## What I took away

- Agents with a good skill file may produce custom GPU code unattended.
- Treat 4x speedups as claims until reproduced.
- Benchmark against a baseline on the same hardware.

## Reference

- Original post: [https://x.com/tensorfish/status/2124975715141689818](https://x.com/tensorfish/status/2124975715141689818)
- Author: [@tensorfish](https://x.com/tensorfish) on X, posted 2026-11-23
