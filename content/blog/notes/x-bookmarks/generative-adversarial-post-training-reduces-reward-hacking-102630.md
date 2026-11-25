---
title: "Generative Adversarial Post-Training Reduces Reward Hacking"
publishDate: "2025-11-28"
date: "2025-11-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-safety"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1994505517063102630-1.png"
excerpt: "A paper applies GAN-style discrimination to RL post-training of LLMs to reduce reward hacking."
---

> Source: [@natashajaques on X](https://x.com/natashajaques/status/1994505517063102630) · posted 2025-11-28

![Generative Adversarial Post-Training Reduces Reward Hacking](/imgs/blogs/x-bookmarks/1994505517063102630-1.png)

## Summary

The author is excited about a paper applying GAN ideas, specifically GAIL, to RL post-training of LLMs. She says it significantly mitigates reward hacking and lets generative music models learn to jam live with people. The quoted post introduces **GAPT**, which adds a GAN-style discriminator on policy rollouts.

## Background

Reward hacking is when a model exploits flaws in the reward instead of doing the real task. A discriminator that tells model outputs from real ones pushes the model to stay realistic, limiting exploitation.

## What I took away

- A discriminator can act as a guard against reward hacking.
- Reward hacking is a central problem in RL post-training.

## Reference

- Original post: [https://x.com/natashajaques/status/1994505517063102630](https://x.com/natashajaques/status/1994505517063102630)
- Author: [@natashajaques](https://x.com/natashajaques) on X, posted 2025-11-28
