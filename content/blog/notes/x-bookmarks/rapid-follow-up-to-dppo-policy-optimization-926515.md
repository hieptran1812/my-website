---
title: "Rapid Follow-Up to DPPO Policy Optimization"
publishDate: "2026-06-09"
date: "2026-06-09"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2064286739875926515-1.jpg"
excerpt: "A post noting a new soft version of DPPO appeared soon after the original, using a divergence mask."
---

> Source: [@sheriyuo on X](https://x.com/sheriyuo/status/2064286739875926515) · posted 2026-06-09

![Rapid Follow-Up to DPPO Policy Optimization (1)](/imgs/blogs/x-bookmarks/2064286739875926515-1.jpg)

![Rapid Follow-Up to DPPO Policy Optimization (2)](/imgs/blogs/x-bookmarks/2064286739875926515-2.jpg)

## Summary

@sheriyuo notes policy optimization is moving fast: DPPO barely reached ICML and a successor exists. The quoted post says the new method is a soft version of DPPO, which uses a divergence-based mask instead of ratio-based clipping.

## Background

PPO-style methods clip the probability ratio to keep updates small. DPPO instead masks based on divergence.

## What I took away

- Policy optimization changes quickly.
- Divergence masks replace ratio clipping.

## Reference

- Original post: [https://x.com/sheriyuo/status/2064286739875926515](https://x.com/sheriyuo/status/2064286739875926515)
- Author: [@sheriyuo](https://x.com/sheriyuo) on X, posted 2026-06-09
