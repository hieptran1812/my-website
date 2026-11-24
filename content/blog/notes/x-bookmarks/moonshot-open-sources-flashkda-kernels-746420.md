---
title: "Moonshot Open Sources FlashKDA Kernels"
publishDate: "2026-07-27"
date: "2026-07-27"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-architecture", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "Moonshot releases FlashKDA, CUTLASS-based Kimi Delta Attention kernels with a claimed 1.72x to 2.22x prefill speedup on H20."
---

> Source: [@Kimi_Moonshot on X](https://x.com/Kimi_Moonshot/status/2081762799202746420) · posted 2026-07-27

## Summary

Moonshot says it open sourced **FlashKDA**, a high-performance CUTLASS-based implementation of Kimi Delta Attention kernels. It claims 1.72x to 2.22x prefill speedup over the flash-linear-attention baseline on H20. It works as a drop-in backend for flash-linear-attention.

## Background

Kimi Delta Attention is a linear attention variant: it keeps a fixed-size state instead of a growing KV cache, which helps long contexts. CUTLASS is NVIDIA's library of templates for writing fast GPU kernels. H20 is an NVIDIA GPU sold in China.

## What I took away

- New attention types need optimized kernels before they are practical.
- Drop-in backends make it easy to adopt faster kernels.
- Compare with later kernels like the Blackwell CAKE ones, which claim further gains.

## Reference

- Original post: [https://x.com/Kimi_Moonshot/status/2081762799202746420](https://x.com/Kimi_Moonshot/status/2081762799202746420)
- Author: [@Kimi_Moonshot](https://x.com/Kimi_Moonshot) on X, posted 2026-07-27
