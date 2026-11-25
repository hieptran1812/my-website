---
title: "Architecture Notes On Inkling 2 Small"
publishDate: "2026-11-12"
date: "2026-11-12"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "quantization", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2121015053206090114-1.jpg"
excerpt: "Comments on the Inkling 2 Small architecture, including mixed NVFP4 and MXFP8 experts and attention changes."
---

> Source: [@eliebakouch on X](https://x.com/eliebakouch/status/2121015053206090114) · posted 2026-11-12

![Architecture Notes On Inkling 2 Small](/imgs/blogs/x-bookmarks/2121015053206090114-1.jpg)

## Summary

The author adds architecture details to a deep dive by Songlin Yang: experts in NVFP4 for layers 0-39 and MXFP8 in the last 8, softcaps in the LM head and SwiGLU, and sliding window attention growing from 512 to 1k. The quoted Thinking Machines post introduces Inkling 2 Small with attention residuals, gated attention, dynamic attention sink, dynamic shortconv and value norm.

## Background

NVFP4 and MXFP8 are low-precision number formats that save memory and compute. Experts are the sub-networks in a mixture-of-experts model. Sliding window attention (SWA) limits each token to nearby context. Mixed precision across layers is a design choice here.

## What I took away

- Precision can vary by layer depth.
- Many small attention tweaks are combined in modern models.
- Read the primary release for exact specs.

## Reference

- Original post: [https://x.com/eliebakouch/status/2121015053206090114](https://x.com/eliebakouch/status/2121015053206090114)
- Author: [@eliebakouch](https://x.com/eliebakouch) on X, posted 2026-11-12
