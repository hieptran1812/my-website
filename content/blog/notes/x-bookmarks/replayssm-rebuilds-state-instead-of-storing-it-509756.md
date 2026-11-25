---
title: "ReplaySSM Rebuilds State Instead of Storing It"
publishDate: "2026-06-15"
date: "2026-06-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "llm-inference", "gpu-kernels"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2066517132733509756-1.jpg"
excerpt: "A post introducing ReplaySSM, which caches recent inputs and rebuilds the SSM state to speed up decoding."
---

> Source: [@zwljohnny on X](https://x.com/zwljohnny/status/2066517132733509756) · posted 2026-06-15

![ReplaySSM Rebuilds State Instead of Storing It](/imgs/blogs/x-bookmarks/2066517132733509756-1.jpg)

## Summary

@zwljohnny asks why the SSM state is stored every step. ReplaySSM caches recent inputs instead and rebuilds the state on the fly, giving the same outputs. The motivation is that more models are hybrids, such as Nemotron-3 and Qwen3.5.

## Background

An SSM (state space model) layer keeps a running state instead of a KV cache. In decoding, writing that state back to memory each step costs bandwidth. Recomputing it from a few cached inputs trades extra compute for less memory traffic, which can be faster on GPUs.

## What I took away

- Decoding is often memory-bound, so recompute can beat storage.
- Hybrid models make SSM speed a real serving concern.

## Reference

- Original post: [https://x.com/zwljohnny/status/2066517132733509756](https://x.com/zwljohnny/status/2066517132733509756)
- Author: [@zwljohnny](https://x.com/zwljohnny) on X, posted 2026-06-15
