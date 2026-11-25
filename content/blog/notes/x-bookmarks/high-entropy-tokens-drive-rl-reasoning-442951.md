---
title: "High-Entropy Tokens Drive RL Reasoning"
publishDate: "2025-12-13"
date: "2025-12-13"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "reinforcement-learning", "post-training", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/1999648926719442951-1.jpg"
excerpt: "A NeurIPS 2025 Qwen paper claims applying the RLVR loss only to the top 20 percent highest-entropy tokens works well."
---

> Source: [@_xjdr on X](https://x.com/_xjdr/status/1999648926719442951) · posted 2025-12-13

![High-Entropy Tokens Drive RL Reasoning](/imgs/blogs/x-bookmarks/1999648926719442951-1.jpg)

## Summary

@_xjdr calls it a good paper and solid concept. The quoted post summarizes Beyond the 80/20 Rule from the Qwen team: in GRPO-like RLVR you should apply the loss only to the 20% highest-entropy tokens.

## Background

RLVR is RL with verifiable rewards. Entropy measures how uncertain the model is about a token. The idea is that the few uncertain tokens act as decision points, so focusing the learning signal there may be enough.

## What I took away

- Not all tokens matter equally for RL.
- Masking the loss to uncertain tokens is a cheap idea to try.
- Verify on your own task.

## Reference

- Original post: [https://x.com/_xjdr/status/1999648926719442951](https://x.com/_xjdr/status/1999648926719442951)
- Author: [@_xjdr](https://x.com/_xjdr) on X, posted 2025-12-13
