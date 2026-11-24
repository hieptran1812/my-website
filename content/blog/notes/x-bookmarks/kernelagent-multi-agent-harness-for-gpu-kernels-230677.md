---
title: "KernelAgent Multi-Agent Harness for GPU Kernels"
publishDate: "2026-10-08"
date: "2026-10-08"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "ai-agents", "harness-engineering"]
author: "Hiep Tran"
featured: false
excerpt: "A PyTorchCon talk presents KernelAgent, a multi-agent system that writes and optimizes GPU kernels."
---

> Source: [@PyTorch on X](https://x.com/PyTorch/status/2108292066153230677) · posted 2026-10-08

## Summary

PyTorch notes that optimizing kernels takes time and expert knowledge. KernelAgent is a multi-agent harness that writes and optimizes kernels for you. Kaiming Cheng from Meta will present it at PyTorchCon North America 2026 in a talk about hardware-guided GPU kernel work.

## Background

A GPU kernel is a small program that runs a specific operation, like matrix multiply, on the GPU. A harness is the scaffolding around an LLM agent: tools, tests, feedback and loop logic.

## What I took away

- Agents with profiler feedback may take over routine kernel tuning.
- Correctness tests are essential when letting agents write kernels.
- Watch for the talk recording.

## Reference

- Original post: [https://x.com/PyTorch/status/2108292066153230677](https://x.com/PyTorch/status/2108292066153230677)
- Author: [@PyTorch](https://x.com/PyTorch) on X, posted 2026-10-08
