---
title: "Pass@N In Multi-Turn Versus Single-Turn RL"
publishDate: "2025-12-20"
date: "2025-12-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2002326974019277248-1.jpg"
excerpt: "Researchers explain why pass@N may increase with training in multi-turn RL, unlike in single-turn reasoning RL."
---

> Source: [@ShengjieWa34067 on X](https://x.com/ShengjieWa34067/status/2002326974019277248) · posted 2025-12-20

![Pass@N In Multi-Turn Versus Single-Turn RL](/imgs/blogs/x-bookmarks/2002326974019277248-1.jpg)

## Summary

The author offers intuition on why pass@N behaves differently in multi-turn and single-turn RL. Prior work on single-turn reasoning models shows RL does not improve pass@N. The quoted post says that in their multi-turn experiments pass@N does increase with training steps.

## Background

Pass@N is the chance that at least one of N samples is correct. If RL only sharpens the model's top answer, pass@N for large N stays flat. If it grows, RL may be teaching new capability rather than only reweighting existing ones.

## What I took away

- Pass@N helps tell whether RL adds new capability or only sharpens.
- Multi-turn settings may behave differently from single-turn ones.
- Watch pass@N, not only pass@1, when evaluating RL.

## Reference

- Original post: [https://x.com/ShengjieWa34067/status/2002326974019277248](https://x.com/ShengjieWa34067/status/2002326974019277248)
- Author: [@ShengjieWa34067](https://x.com/ShengjieWa34067) on X, posted 2025-12-20
