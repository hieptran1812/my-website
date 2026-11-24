---
title: "What Sliding Window Attention Is"
publishDate: "2026-07-16"
date: "2026-07-16"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "kv-cache", "learning-resources"]
author: "Hiep Tran"
featured: false
excerpt: "An explainer of sliding window attention, prompted by the observation that many western open models use it."
---

> Source: [@NielsRogge on X](https://x.com/NielsRogge/status/2077698690542653443) · posted 2026-07-16

## Summary

The quoted post notes that open models from western big labs, such as Thinking Machines, Microsoft MAI, Gemma and gpt-oss, all use sliding window attention. The author explains that SWA restricts each token to attending only within a fixed-size neighborhood instead of the full sequence. This reduces attention compute and KV-cache memory for long-context models.

## Background

Full attention costs grow with sequence length, and the KV cache grows linearly. SWA caps how far back each token looks, so memory per layer stays fixed. Models usually mix SWA layers with some full-attention layers to keep long-range ability.

## What I took away

- SWA is now a standard tool for long-context efficiency.
- Hybrid designs mix local and global layers to balance cost and recall.
- Serving systems need SWA-aware caches to realize the savings.

## Reference

- Original post: [https://x.com/NielsRogge/status/2077698690542653443](https://x.com/NielsRogge/status/2077698690542653443)
- Author: [@NielsRogge](https://x.com/NielsRogge) on X, posted 2026-07-16
