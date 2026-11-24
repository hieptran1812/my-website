---
title: "Making Inference Cheaper With LMCache KV Reuse"
publishDate: "2026-07-11"
date: "2026-07-11"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "kv-cache", "llm-inference", "serving"]
author: "Hiep Tran"
featured: false
excerpt: "A promotional summary of a talk on cutting LLM inference cost by reusing KV cache through LMCache."
---

> Source: [@h100envy on X](https://x.com/h100envy/status/2075962957649899829) · posted 2026-07-11

## Summary

The author says an ex-vLLM core contributor explained how to make LLM inference 10x cheaper in 34 minutes. The loop: a request arrives, check LMCache, and on a hit load the KV cache from CPU, SSD or remote storage, skip prefill and serve. The post says this loop explains why companies like Bloomberg use it; the bootcamp comparison and 10x figure are the author's framing.

## Background

Prefill is the step where the model processes the whole prompt to build the KV cache, which is costly for long prompts. If the same prefix was seen before, its cache can be stored and reloaded instead of recomputed. LMCache is a tool that stores KV caches across memory tiers.

## What I took away

- Reusing KV cache helps most when many requests share long prefixes.
- Loading cache from slower storage must be faster than recomputing to pay off.
- Treat '10x cheaper' as workload-dependent.

## Reference

- Original post: [https://x.com/h100envy/status/2075962957649899829](https://x.com/h100envy/status/2075962957649899829)
- Author: [@h100envy](https://x.com/h100envy) on X, posted 2026-07-11
