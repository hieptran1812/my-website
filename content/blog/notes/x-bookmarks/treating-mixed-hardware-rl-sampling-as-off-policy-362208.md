---
title: "Treating Mixed-Hardware RL Sampling as Off-Policy"
publishDate: "2026-10-13"
date: "2026-10-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "hardware"]
author: "Hiep Tran"
featured: false
excerpt: "Training with RL samplers on different precisions or accelerators can be handled with standard off-policy correction."
---

> Source: [@nanjiangwill on X](https://x.com/nanjiangwill/status/2110085048716362208) · posted 2026-10-13

## Summary

The author says that a sampler running on a different precision or accelerator than the trainer is simply off-policy, like stale samples in fully asynchronous RL. Off-policy correction does not care why the sampler differs, so the same fix applies. The quoted post explains the motivation: RL for large models usually runs on one fixed cluster of the latest accelerators for stability, which wastes inference work done for other vendors and older hardware and limits usable capacity.

## Background

In RL for LLMs, a sampler generates rollouts and a trainer updates the weights. If the sampler's outputs come from a slightly different distribution than the current policy, the data is off-policy, and methods like importance weighting correct for the mismatch.

## What I took away

- Numerical differences between inference and training stacks can be framed as an off-policy problem.
- This framing could let RL jobs use heterogeneous or older GPUs.
- Look for the size of the correction weights; large mismatches still hurt.

## Reference

- Original post: [https://x.com/nanjiangwill/status/2110085048716362208](https://x.com/nanjiangwill/status/2110085048716362208)
- Author: [@nanjiangwill](https://x.com/nanjiangwill) on X, posted 2026-10-13
