---
title: "Faster Sampling Kernels and Inference Revenue"
publishDate: "2026-07-26"
date: "2026-07-26"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "gpu-kernels", "llm-inference"]
author: "Hiep Tran"
featured: false
excerpt: "The author argues SonicSampler-style work could let providers generate 15% more tokens and so earn 15% more revenue."
---

> Source: [@yifanzhang_ on X](https://x.com/yifanzhang_/status/2081454255437996541) · posted 2026-07-26

## Summary

The author says he is optimistic about tokenomics. He claims this line of work can enable 15% more tokens, which he says translates to 15% more revenue. The quoted post is his earlier note on SonicSampler from Tri Dao's team at Together AI.

## Background

Tokenomics here means the economics of serving tokens: cost per token and revenue per GPU. If kernels make generation faster on the same hardware, a provider serves more tokens for the same cost. The 15% figure is the author's claim.

## What I took away

- Small kernel speedups add up to real money at scale.
- Revenue gains also depend on demand, not just capacity.
- Treat the 15% figure as a claim to check against the paper.

## Reference

- Original post: [https://x.com/yifanzhang_/status/2081454255437996541](https://x.com/yifanzhang_/status/2081454255437996541)
- Author: [@yifanzhang_](https://x.com/yifanzhang_) on X, posted 2026-07-26
