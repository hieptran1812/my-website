---
title: "SonicSampler Fuses LLM Sampling and Speculative Verification"
publishDate: "2026-07-26"
date: "2026-07-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "speculative-decoding", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "Together AI's SonicSampler paper introduces tile-aware kernels for LLM sampling and speculative verification, building on FlashSampling."
---

> Source: [@yifanzhang_ on X](https://x.com/yifanzhang_/status/2081447805303058447) · posted 2026-07-26

## Summary

The author highlights **SonicSampler**: Unified Tile-Aware Kernels for LLM Sampling and Speculative Verification, from Tri Dao's team at Together AI, with an arXiv link. He says it brings to life the future work described in his team's FlashSampling paper. He congratulates the team.

## Background

Sampling is choosing the next token from the model's probability distribution, often with top-k or top-p filters. Speculative verification checks drafted tokens against the main model. Doing these steps in fast fused GPU kernels avoids extra memory trips, which matters with large vocabularies.

## What I took away

- Sampling is a non-trivial cost at scale and can be optimized.
- Tile-aware kernels process data in chunks that fit GPU memory well.
- Read both FlashSampling and SonicSampler to see how the idea evolved.

## Reference

- Original post: [https://x.com/yifanzhang_/status/2081447805303058447](https://x.com/yifanzhang_/status/2081447805303058447)
- Author: [@yifanzhang_](https://x.com/yifanzhang_) on X, posted 2026-07-26
