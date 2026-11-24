---
title: "qwenGPU Runs CUDA Inside Model Weights"
publishDate: "2026-11-02"
date: "2026-11-02"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "gpu-kernels", "interpretability"]
author: "Hiep Tran"
featured: false
excerpt: "A project compiles a SIMT GPU into Qwen3.5-2B's weights so the model runs real CUDA in its forward passes."
---

> Source: [@mike64_t on X](https://x.com/mike64_t/status/2117134542519799969) · posted 2026-11-02

## Summary

The quoted post introduces qwenGPU: Qwen3.5-2B with a SIMT GPU inside its weights, adding 45 compiled layers in front of Qwen's own and using its Gated DeltaNet layers as registers and memory. It runs real CUDA in its forward passes; the host compiles the kernel to PTX and loads it, then the model executes it. @mike64_t reacts with amazement.

## Background

SIMT (single instruction, multiple threads) is how GPUs run the same instruction across many threads. PTX is NVIDIA's low-level intermediate code. Compiling this into weights shows a transformer can emulate a processor when its weights are set by hand.

## What I took away

- Neural networks can be hand-built to emulate real hardware.
- Recurrent-state layers like Gated DeltaNet can act as memory.
- These projects teach a lot about what layers can compute exactly.

## Reference

- Original post: [https://x.com/mike64_t/status/2117134542519799969](https://x.com/mike64_t/status/2117134542519799969)
- Author: [@mike64_t](https://x.com/mike64_t) on X, posted 2026-11-02
