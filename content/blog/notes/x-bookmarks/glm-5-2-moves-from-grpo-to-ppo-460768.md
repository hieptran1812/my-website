---
title: "GLM-5.2 Moves From GRPO To PPO"
publishDate: "2026-06-20"
date: "2026-06-20"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "open-models", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2068395760509460768-1.jpg"
excerpt: "A post notes GLM-5.2 switches from GRPO to PPO for long-horizon agentic RL and quotes the GLM-5.2 announcement."
---

> Source: [@burny_tech on X](https://x.com/burny_tech/status/2068395760509460768) · posted 2026-06-20

![GLM-5.2 Moves From GRPO To PPO (1)](/imgs/blogs/x-bookmarks/2068395760509460768-1.jpg)

![GLM-5.2 Moves From GRPO To PPO (2)](/imgs/blogs/x-bookmarks/2068395760509460768-2.jpg)

## Summary

@burny_tech says GLM-5.2 moves from GRPO to PPO in long-horizon agentic RL, noting DeepSeek's GRPO paper showed it can match PPO without a critic. The quoted GLM-5.2 announcement lists coding and agentic gains, a 1M context window, and two reasoning levels, max and high.

## Background

GRPO estimates advantage from a group of samples, while PPO uses a critic. The switch suggests critics help for very long tasks. 1M context means the model reads up to about a million tokens.

## What I took away

- Frontier labs are revisiting PPO for long-horizon RL.
- Critic-free is not always enough.

## Reference

- Original post: [https://x.com/burny_tech/status/2068395760509460768](https://x.com/burny_tech/status/2068395760509460768)
- Author: [@burny_tech](https://x.com/burny_tech) on X, posted 2026-06-20
