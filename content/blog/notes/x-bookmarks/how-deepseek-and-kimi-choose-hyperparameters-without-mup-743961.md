---
title: "How DeepSeek And Kimi Choose Hyperparameters Without muP"
publishDate: "2026-06-15"
date: "2026-06-15"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "pretraining", "learning-resources", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2066615581642743961-1.jpg"
excerpt: "A post explains DeepSeek v1 to v3 and K2 skip muP and use naive initialization, fitting scaling laws for batch size and learning rate."
---

> Source: [@nrehiew_ on X](https://x.com/nrehiew_/status/2066615581642743961) · posted 2026-06-15

![How DeepSeek And Kimi Choose Hyperparameters Without muP (1)](/imgs/blogs/x-bookmarks/2066615581642743961-1.jpg)

![How DeepSeek And Kimi Choose Hyperparameters Without muP (2)](/imgs/blogs/x-bookmarks/2066615581642743961-2.jpg)

![How DeepSeek And Kimi Choose Hyperparameters Without muP (3)](/imgs/blogs/x-bookmarks/2066615581642743961-3.jpg)

## Summary

@nrehiew_ says this content is free. The quoted post says DeepSeek v1 to v3 (no detail in v4) and K2 do not use muP, using naive N(0, 0.006) initialization. They select hyperparameters by fitting scaling laws to get optimal batch size and learning rate.

## Background

muP lets hyperparameters transfer across model sizes. Instead, these labs fit scaling laws on small runs to predict good batch size and learning rate at scale.

## What I took away

- Scaling-law fits can replace muP.
- Hyperparameters are chosen empirically.

## Reference

- Original post: [https://x.com/nrehiew_/status/2066615581642743961](https://x.com/nrehiew_/status/2066615581642743961)
- Author: [@nrehiew_](https://x.com/nrehiew_) on X, posted 2026-06-15
