---
title: "SFT, RL and OPD as Distribution Reshaping"
publishDate: "2026-06-15"
date: "2026-06-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "post-training", "reinforcement-learning", "distillation"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2066427407146643561-1.jpg"
excerpt: "A post praising a blog that frames SFT, RL and on-policy distillation as ways to reshape a model's distribution."
---

> Source: [@liulicheng10 on X](https://x.com/liulicheng10/status/2066427407146643561) · posted 2026-06-15

![SFT, RL and OPD as Distribution Reshaping](/imgs/blogs/x-bookmarks/2066427407146643561-1.jpg)

## Summary

@liulicheng10 calls it probably the best blog in a while. Viewing SFT, RL and OPD as different ways of reshaping a distribution makes tradeoffs intuitive: SFT pulls toward a fixed external target, RL moves along the reward gradient on on-policy samples, and OPD is cut off in the post.

## Background

SFT imitates fixed data, RL learns from rewards on its own samples, and OPD (on-policy distillation) has a teacher grade the student's own samples. One shared view makes comparing them easier.

## What I took away

- Think of post-training methods as shaping a distribution.
- Ask whose samples the method trains on.

## Reference

- Original post: [https://x.com/liulicheng10/status/2066427407146643561](https://x.com/liulicheng10/status/2066427407146643561)
- Author: [@liulicheng10](https://x.com/liulicheng10) on X, posted 2026-06-15
