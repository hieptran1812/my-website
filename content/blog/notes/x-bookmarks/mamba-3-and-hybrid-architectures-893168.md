---
title: "Mamba-3 And Hybrid Architectures"
publishDate: "2026-03-18"
date: "2026-03-18"
category: "notes"
subcategory: "X Bookmarks"
tags: ["x-bookmarks", "llm-architecture", "ai-research"]
author: "Hiep Tran"
featured: false
image: "/imgs/blogs/x-bookmarks/2034088726997893168-1.jpg"
excerpt: "Sebastian Raschka reacts to Mamba-3 and suggests swapping it into transformer hybrids in place of Gated DeltaNet."
---

> Source: [@rasbt on X](https://x.com/rasbt/status/2034088726997893168) · posted 2026-03-18

![Mamba-3 And Hybrid Architectures (1)](/imgs/blogs/x-bookmarks/2034088726997893168-1.jpg)

![Mamba-3 And Hybrid Architectures (2)](/imgs/blogs/x-bookmarks/2034088726997893168-2.jpg)

## Summary

The author says Mamba-3 is here and that the most interesting use of Mamba-like models is in hybrids with attention, such as Qwen3.5 and Kimi Linear. They would like to see Gated DeltaNet swapped for Mamba-3, which now also has RoPE. The quoted post announces Mamba-3, designed for the next generation of linear models.

## Background

Mamba is a state space model (SSM) that processes sequences with a fixed-size state, so cost grows slowly with length. Hybrids mix these layers with attention. RoPE is a way of encoding token position.

## What I took away

- Hybrid linear plus attention models are a leading architecture trend.
- Swapping the linear layer in a hybrid is an easy experiment.
- Mamba-3 adds RoPE.

## Reference

- Original post: [https://x.com/rasbt/status/2034088726997893168](https://x.com/rasbt/status/2034088726997893168)
- Author: [@rasbt](https://x.com/rasbt) on X, posted 2026-03-18
