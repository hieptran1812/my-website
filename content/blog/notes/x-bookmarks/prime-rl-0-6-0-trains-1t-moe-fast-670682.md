---
title: "prime-rl 0.6.0 Trains 1T MoE Fast"
publishDate: "2026-06-23"
date: "2026-06-23"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "moe", "serving"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2069253368770670682-1.jpg"
excerpt: "prime-rl v0.6.0 claims RL on a 1T-parameter MoE in under 5 minutes per step, or about 1k steps in 3 days."
---

> Source: [@samsja19 on X](https://x.com/samsja19/status/2069253368770670682) · posted 2026-06-23

![prime-rl 0.6.0 Trains 1T MoE Fast](/imgs/blogs/x-bookmarks/2069253368770670682-1.jpg)

## Summary

@samsja19 says prime-rl can train 1T parameter MoE models in under 5 minutes per step, 1k steps in about 3 days. Inference changes include wide-ep, fp8, llm-d router, mooncake and KV cache CPU offloading, plus fsdp2 and deep-ep for training. The quoted post says GLM-5 on agentic SWE tasks at 131k context reaches sub-5-minute steps.

## Background

RL for LLMs alternates between generating rollouts and updating weights, so both inference and training must be fast. Expert parallelism spreads MoE experts across GPUs. The speed numbers are the authors' claims.

## What I took away

- RL at 1T scale needs optimized inference and training together.
- Step time is the key metric for RL infra.

## Reference

- Original post: [https://x.com/samsja19/status/2069253368770670682](https://x.com/samsja19/status/2069253368770670682)
- Author: [@samsja19](https://x.com/samsja19) on X, posted 2026-06-23
