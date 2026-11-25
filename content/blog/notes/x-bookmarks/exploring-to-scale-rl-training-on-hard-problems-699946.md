---
title: "Exploring To Scale RL Training On Hard Problems"
publishDate: "2025-11-28"
date: "2025-11-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1994409862344699946-1.jpg"
excerpt: "A CMU blog post on how to explore so RL training of LLMs can learn from hard problems."
---

> Source: [@rsalakhu on X](https://x.com/rsalakhu/status/1994409862344699946) · posted 2025-11-28

![Exploring To Scale RL Training On Hard Problems](/imgs/blogs/x-bookmarks/1994409862344699946-1.jpg)

## Summary

The author shares an ML@CMU blog post titled How to Explore to Scale RL Training of LLMs on Hard Problems. They say current on-policy RL methods fail on hard problems because they rarely produce a single correct rollout, giving no reward signal. The text is cut off after that.

## Background

In RL for LLMs, the model tries answers and is rewarded for correct ones. If the problem is too hard, every attempt fails, so there is nothing to learn from. Better exploration is needed to find at least some correct attempts.

## What I took away

- No successful rollouts means no learning signal.
- Exploration strategy matters as models face harder tasks.

## Reference

- Original post: [https://x.com/rsalakhu/status/1994409862344699946](https://x.com/rsalakhu/status/1994409862344699946)
- Author: [@rsalakhu](https://x.com/rsalakhu) on X, posted 2025-11-28
