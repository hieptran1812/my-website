---
title: "Qwen3.6-27B NVFP4-MTP Versus XS Variant"
publishDate: "2026-07-03"
date: "2026-07-03"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "quantization", "open-models"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2072905660090462282-1.jpg"
excerpt: "A post compares two quantized builds of Qwen3.6-27B, finding the full NVFP4-MTP clearly beats the XS one."
---

> Source: [@AgentSparko on X](https://x.com/AgentSparko/status/2072905660090462282) · posted 2026-07-03

![Qwen3.6-27B NVFP4-MTP Versus XS Variant](/imgs/blogs/x-bookmarks/2072905660090462282-1.jpg)

## Summary

@AgentSparko reports a composite score of 86.0 versus 74.7, with math 100% vs 82%, prose 90% vs 73% and coding 80% vs 52%, for AEON-7's NVFP4-MTP over the XS version. The quoted post says the earlier guides pointed to XS for size and speed, but this one is worth switching to for quality.

## Background

NVFP4 is a 4-bit float format. MTP means multi-token prediction heads that help speculative decoding. A smaller variant saves memory but can cost quality.

## What I took away

- Smaller quantized builds can lose significant quality.
- Benchmark variants on your task.
- Scores come from a single informal test.

## Reference

- Original post: [https://x.com/AgentSparko/status/2072905660090462282](https://x.com/AgentSparko/status/2072905660090462282)
- Author: [@AgentSparko](https://x.com/AgentSparko) on X, posted 2026-07-03
