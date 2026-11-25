---
title: "Genshin Gameplay Agent Trained With SFT Only"
publishDate: "2025-11-28"
date: "2025-11-28"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "ai-research", "post-training", "ai-agents"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1994202848574234797-1.png"
excerpt: "A quoted post describes a ByteDance Seed paper that trained Qwen2VL-7B to play Genshin with supervised learning."
---

> Source: [@zephyr_z9 on X](https://x.com/zephyr_z9/status/1994202848574234797) · posted 2025-11-28

![Genshin Gameplay Agent Trained With SFT Only (1)](/imgs/blogs/x-bookmarks/1994202848574234797-1.png)

![Genshin Gameplay Agent Trained With SFT Only (2)](/imgs/blogs/x-bookmarks/1994202848574234797-2.jpg)

## Summary

The author asks how they missed this and tags teortaxesTex. The quoted post says a ByteDance Seed paper trained Qwen2VL-7B to play Genshin using SFT only, no RL, with 2424 hours of human gameplay and 15k short reasoning traces, in under 20k H100 hours over 3 epochs. The rest is cut off.

## Background

SFT (supervised fine-tuning) trains a model to imitate example data, here human gameplay. Many assume RL is needed for agent skills, so a result with SFT only is notable. Qwen2VL-7B is a 7-billion-parameter vision-language model.

## What I took away

- Large amounts of demonstration data can train capable agents without RL.
- The reported compute is under 20k H100 hours.
- Check the paper for evaluation details.

## Reference

- Original post: [https://x.com/zephyr_z9/status/1994202848574234797](https://x.com/zephyr_z9/status/1994202848574234797)
- Author: [@zephyr_z9](https://x.com/zephyr_z9) on X, posted 2025-11-28
