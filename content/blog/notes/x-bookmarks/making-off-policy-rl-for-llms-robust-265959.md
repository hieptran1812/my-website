---
title: "Making Off-Policy RL For LLMs Robust"
publishDate: "2025-12-05"
date: "2025-12-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1996769053420265959-1.jpg"
excerpt: "A study of off-policy RL for LLMs, where TBA stays stable while Dr. GRPO collapses at 10 steps off-policy."
---

> Source: [@bartoldson on X](https://x.com/bartoldson/status/1996769053420265959) · posted 2025-12-05

![Making Off-Policy RL For LLMs Robust](/imgs/blogs/x-bookmarks/1996769053420265959-1.jpg)

## Summary

@bartoldson says off-policy RL for LLMs is hard: Dr. GRPO collapses at 10 steps off-policy, but TBA does not. They say Kimi K2's approach is also robust and that both independently landed on the same key ingredients. The work ablates RL recipe ingredients and shows two small changes that give off-policy robustness.

## Background

On-policy RL needs fresh samples from the current model. Off-policy means training on samples from older versions, which is more efficient for large systems but can destabilize training. GRPO is a common RL algorithm for LLMs.

## What I took away

- Off-policy training can speed up RL systems but needs a careful recipe.
- Two independent groups reaching the same ingredients suggests those ingredients matter.

## Reference

- Original post: [https://x.com/bartoldson/status/1996769053420265959](https://x.com/bartoldson/status/1996769053420265959)
- Author: [@bartoldson](https://x.com/bartoldson) on X, posted 2025-12-05
