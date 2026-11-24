---
title: "Together Compiler for Continual Inference Optimization"
publishDate: "2026-11-05"
date: "2026-11-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "ai-agents"]
author: "Hiep Tran"
featured: false
excerpt: "Together AI shares a compiler that uses an agent loop to continually optimize kernels and inference for shifting production workloads."
---

> Source: [@tri_dao on X](https://x.com/tri_dao/status/2118418321456660572) · posted 2026-11-05

## Summary

Together AI introduced the Together Compiler, described as continual compilation for AI inference. They say inference today is polymorphic, constantly changing, and note that on OpenRouter 80% of traffic goes to recently released models. @tri_dao says the project distills most of their kernel and inference-engine expertise into an agent loop that powers production, after two years of work.

## Background

Inference speed depends on hand-tuned GPU kernels, small programs that run math on the GPU. When new models and input shapes appear constantly, hand-tuning cannot keep up. An agent loop that keeps generating and testing optimized code lets the system adapt as traffic changes.

## What I took away

- Inference workloads change fast, so static one-time optimization goes stale.
- Encoding expert kernel knowledge into an automated agent loop is a pattern worth studying.
- New models dominate traffic, so time-to-optimized-serving matters.

## Reference

- Original post: [https://x.com/tri_dao/status/2118418321456660572](https://x.com/tri_dao/status/2118418321456660572)
- Author: [@tri_dao](https://x.com/tri_dao) on X, posted 2026-11-05
