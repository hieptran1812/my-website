---
title: "Zero-Overhead Multi-Layer MTP For MiMo In SGLang"
publishDate: "2025-12-16"
date: "2025-12-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "speculative-decoding", "serving", "open-models"]
author: "Hiep Tran"
featured: false
excerpt: "ying11231 congratulates Xiaomi MiMo and notes SGLang's day-0 support with zero-overhead multi-layer MTP."
---

> Source: [@ying11231 on X](https://x.com/ying11231/status/2000979825801203730) · posted 2025-12-16

## Summary

The author congratulates the MiMo team on a model designed for efficient inference with good agentic performance, and highlights zero-overhead multi-layer MTP on SGLang's spec V2 design. The quoted post announces SGLang's day-0 support for MiMo-V2-Flash, an inference-centric model with sliding-window attention and multi-layer MTP.

## Background

MTP (multi-token prediction) trains extra heads that predict several future tokens, which can then be used for speculative decoding: draft tokens are checked by the main model in one pass. Zero overhead means the draft and verification scheduling adds almost no extra cost.

## What I took away

- MTP heads can act as built-in drafters for speculative decoding.
- Scheduling overhead can erase speedups, so engine design matters.
- Models built for inference speed need matching serving support.

## Reference

- Original post: [https://x.com/ying11231/status/2000979825801203730](https://x.com/ying11231/status/2000979825801203730)
- Author: [@ying11231](https://x.com/ying11231) on X, posted 2025-12-16
