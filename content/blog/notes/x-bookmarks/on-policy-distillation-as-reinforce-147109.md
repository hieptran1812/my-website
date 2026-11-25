---
title: "On-Policy Distillation as REINFORCE"
publishDate: "2026-06-05"
date: "2026-06-05"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "distillation", "reinforcement-learning", "post-training"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2062951493960147109-1.png"
excerpt: "A post framing OPD as REINFORCE with a teacher-student log-likelihood advantage, quoting Self-Distilled Policy Gradient."
---

> Source: [@yifanzhang_ on X](https://x.com/yifanzhang_/status/2062951493960147109) · posted 2026-06-05

![On-Policy Distillation as REINFORCE (1)](/imgs/blogs/x-bookmarks/2062951493960147109-1.png)

![On-Policy Distillation as REINFORCE (2)](/imgs/blogs/x-bookmarks/2062951493960147109-2.jpg)

## Summary

@yifanzhang_ says OPD gives dense token-level rewards and process supervision, and can be seen as REINFORCE with advantage equal to the centered teacher-to-student log-likelihood ratio. The quote introduces Self-Distilled Policy Gradient.

## Background

REINFORCE is a basic policy gradient method. Token-level advantages give finer credit than a single reward.

## What I took away

- RL and distillation converge.
- Dense rewards help credit assignment.

## Reference

- Original post: [https://x.com/yifanzhang_/status/2062951493960147109](https://x.com/yifanzhang_/status/2062951493960147109)
- Author: [@yifanzhang_](https://x.com/yifanzhang_) on X, posted 2026-06-05
