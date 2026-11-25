---
title: "LoRA Forgets Less During Multi-Task RL"
publishDate: "2025-12-14"
date: "2025-12-14"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2000110693610385634-1.jpg"
excerpt: "The author observes that with LoRA you can learn a lot in RL while forgetting less."
---

> Source: [@FanqingMengAI on X](https://x.com/FanqingMengAI/status/2000110693610385634) · posted 2025-12-14

![LoRA Forgets Less During Multi-Task RL](/imgs/blogs/x-bookmarks/2000110693610385634-1.jpg)

## Summary

@FanqingMengAI has further observations and says that with LoRA you can learn as much as you want in RL, but forget less. The quoted post says the original plan was to train expert models per sub-task, merge them, and then run multi-task RL.

## Background

LoRA fine-tunes small low-rank add-on weights instead of the whole model. Model merging combines several trained models into one. Forgetting means a model loses earlier skills while learning new ones. These are early observations, not a full study.

## What I took away

- LoRA may reduce forgetting during sequential or multi-task training.
- Merging experts then continuing RL is a pipeline worth testing.
- Treat these as preliminary experiments.

## Reference

- Original post: [https://x.com/FanqingMengAI/status/2000110693610385634](https://x.com/FanqingMengAI/status/2000110693610385634)
- Author: [@FanqingMengAI](https://x.com/FanqingMengAI) on X, posted 2025-12-14
