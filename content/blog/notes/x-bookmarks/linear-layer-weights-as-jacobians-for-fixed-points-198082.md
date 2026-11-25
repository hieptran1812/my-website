---
title: "Linear Layer Weights As Jacobians For Fixed Points"
publishDate: "2025-12-18"
date: "2025-12-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "math", "ai-research", "llm-architecture"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2001452311500198082-1.png"
excerpt: "BetaTomorrow notes that nn.Linear weights act as a dense Jacobian that suits fixed point iteration."
---

> Source: [@BetaTomorrow on X](https://x.com/BetaTomorrow/status/2001452311500198082) · posted 2025-12-18

![Linear Layer Weights As Jacobians For Fixed Points (1)](/imgs/blogs/x-bookmarks/2001452311500198082-1.png)

![Linear Layer Weights As Jacobians For Fixed Points (2)](/imgs/blogs/x-bookmarks/2001452311500198082-2.png)

## Summary

The author says the weight in nn.Linear acts as the Jacobian of that isolated linear layer, which is a full dense matrix. They argue that such a full matrix is ideal for fixed point iteration and call it a hidden power of neural networks. They say it was not mentioned in the part 2 paper but needs more attention.

## Background

A Jacobian is the matrix of how outputs change with inputs. Fixed point iteration repeatedly applies a function until the value stops changing, which is used in solving equations and in some parallel decoding ideas. For a linear layer the Jacobian is just its weight matrix.

## What I took away

- For a linear layer, the Jacobian is simply the weight matrix.
- Dense Jacobians make fixed point methods well behaved.
- Look at the series for the full argument.

## Reference

- Original post: [https://x.com/BetaTomorrow/status/2001452311500198082](https://x.com/BetaTomorrow/status/2001452311500198082)
- Author: [@BetaTomorrow](https://x.com/BetaTomorrow) on X, posted 2025-12-18
