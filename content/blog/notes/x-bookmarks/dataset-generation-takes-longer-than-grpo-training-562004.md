---
title: "Dataset Generation Takes Longer Than GRPO Training"
publishDate: "2025-12-13"
date: "2025-12-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "reinforcement-learning", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999896174577562004-1.jpg"
excerpt: "Authors share that building a good synthetic dataset for GRPO training takes much more time than the training itself."
---

> Source: [@serious_mehta on X](https://x.com/serious_mehta/status/1999896174577562004) · posted 2025-12-13

![Dataset Generation Takes Longer Than GRPO Training](/imgs/blogs/x-bookmarks/1999896174577562004-1.jpg)

## Summary

@serious_mehta says synthetic data cleaning will take at least another week. The quoted post says generating a high-quality dataset for GRPO training is far more time-consuming than the training, with about 12 hours spent and under 40% done.

## Background

GRPO is an RL method for LLMs that needs many good prompts and checkable answers. Making and cleaning that data is often the real bottleneck. Compute for training is only a part of the work.

## What I took away

- Budget more time for data than for training runs.
- Synthetic data needs cleaning and validation.
- Plan data pipelines early in an RL project.

## Reference

- Original post: [https://x.com/serious_mehta/status/1999896174577562004](https://x.com/serious_mehta/status/1999896174577562004)
- Author: [@serious_mehta](https://x.com/serious_mehta) on X, posted 2025-12-13
