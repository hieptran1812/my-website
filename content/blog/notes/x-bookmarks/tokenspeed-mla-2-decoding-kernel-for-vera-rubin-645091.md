---
title: "TokenSpeed MLA 2 Decoding Kernel for Vera Rubin"
publishDate: "2026-10-16"
date: "2026-10-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "TokenSpeed releases MLA 2, a faster multi-head latent attention decode kernel for NVIDIA Vera Rubin, with speedups for Kimi K3."
---

> Source: [@zhyncs42 on X](https://x.com/zhyncs42/status/2110888974063645091) · posted 2026-10-16

## Summary

@zhyncs42 compares TokenSpeed's kernel versions to FlashAttention's: MLA 1 targets Blackwell and MLA 2 targets Vera Rubin, with over 2M monthly PyPI downloads, thanking NVIDIA for early access. The quoted post introduces TokenSpeed MLA 2 and claims Kimi K3 decode kernel speedups of up to 3.44x vs MLA 1 on B200 and up to 2.84x on GB200, combining hardware and software gains.

## Background

MLA (multi-head latent attention) compresses attention keys and values into a smaller latent vector to shrink the KV cache. A decode kernel computes attention for each new token during generation. Kernels are tuned for each GPU generation, so new hardware brings new versions.

## What I took away

- Attention kernels must be retuned for each new GPU generation.
- MLA reduces memory per token, which helps long contexts and big batches.
- Speedup figures mix hardware and software gains.

## Reference

- Original post: [https://x.com/zhyncs42/status/2110888974063645091](https://x.com/zhyncs42/status/2110888974063645091)
- Author: [@zhyncs42](https://x.com/zhyncs42) on X, posted 2026-10-16
