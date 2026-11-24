---
title: "Flash-MSA Open Source Sparse Attention Training Kernels"
publishDate: "2026-07-12"
date: "2026-07-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-architecture", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Flash-MSA releases open source sparse attention training kernels claiming over 4x speedup versus dense FlashAttention at long context."
---

> Source: [@rawsh0 on X](https://x.com/rawsh0/status/2076354449707356558) · posted 2026-07-12

## Summary

The quoted post introduces Flash-MSA, described as the first open source sparse attention training kernels optimized for extreme context lengths: 4 simple kernels with a 400%+ speedup over dense flash attention at long context. The author adds that Ganesh wrote them as a side project on rented GPUs and that they reach 4x+ over dense FlashAttention at 1M context.

## Background

Sparse attention lets each token attend only to selected positions instead of all, cutting cost at long lengths. FlashAttention is the standard fast exact attention kernel. Training kernels also need backward passes, which are harder to write than inference kernels.

## What I took away

- Speedups from sparse attention grow with context length; check at your target length.
- Open training kernels lower the barrier for long-context research.
- Strong kernel work can come from individuals on rented hardware.

## Reference

- Original post: [https://x.com/rawsh0/status/2076354449707356558](https://x.com/rawsh0/status/2076354449707356558)
- Author: [@rawsh0](https://x.com/rawsh0) on X, posted 2026-07-12
