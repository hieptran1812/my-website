---
title: "New Work on Bidirectional Evolutionary Search"
publishDate: "2026-05-28"
date: "2026-05-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2060025514853617678-1.jpg"
excerpt: "Yu Huang shares a colleague's paper on bidirectional evolutionary search for hard reasoning problems."
---

> Source: [@Huangyu58589918 on X](https://x.com/Huangyu58589918/status/2060025514853617678) · posted 2026-05-28

![New Work on Bidirectional Evolutionary Search](/imgs/blogs/x-bookmarks/2060025514853617678-1.jpg)

## Summary

The author promotes new work on bidirectional evolutionary search led by Kevin Xu. The quoted post asks how LLMs should sample on hard reasoning problems where direct rollouts rarely give a correct answer. It notes that Best-of-N and tree search have sparse verification signals and stay within the model's distribution.

## Background

Best-of-N samples many answers and keeps the best. Hard problems may have zero correct samples, so no signal. Evolutionary search tries to improve candidates step by step to reach one.

## What I took away

- Sampling strategy matters for RL on hard problems.
- Credit goes to the lead author; read the thread for method details.
- Look at how it differs from GRPO-style sampling.

## Reference

- Original post: [https://x.com/Huangyu58589918/status/2060025514853617678](https://x.com/Huangyu58589918/status/2060025514853617678)
- Author: [@Huangyu58589918](https://x.com/Huangyu58589918) on X, posted 2026-05-28
