---
title: "Stable GRPO For Multi-Turn Agentic Search"
publishDate: "2025-12-24"
date: "2025-12-24"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "ai-agents", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2003900947488227381-1.jpg"
excerpt: "A post hypes a paper that raised small Qwen3 models from 1-2 search turns to 15+ turns using GRPO."
---

> Source: [@byebyescaling on X](https://x.com/byebyescaling/status/2003900947488227381) · posted 2025-12-24

![Stable GRPO For Multi-Turn Agentic Search (1)](/imgs/blogs/x-bookmarks/2003900947488227381-1.jpg)

![Stable GRPO For Multi-Turn Agentic Search (2)](/imgs/blogs/x-bookmarks/2003900947488227381-2.jpg)

## Summary

The author claims 10x search depth via GRPO and says the problem is the stability of the objective, not compute. They say most 8B models suffer horizon collapse. The quoted post shares engineering practices that lifted Qwen3 8B and Qwen3 A3B from 1-2 search turns and 10% accuracy on Browsecomp-Plus to 15+ and 20+ turns with 30% accuracy.

## Background

GRPO is a reinforcement learning method for training language models using group-based rewards. Agentic search means a model issues many search queries before answering. Horizon collapse here means the model stops after only a few turns. The numbers are the authors' claims.

## What I took away

- For long agent tasks, stable training objectives matter.
- Details of implementation decide whether RL scales to many turns.

## Reference

- Original post: [https://x.com/byebyescaling/status/2003900947488227381](https://x.com/byebyescaling/status/2003900947488227381)
- Author: [@byebyescaling](https://x.com/byebyescaling) on X, posted 2025-12-24
